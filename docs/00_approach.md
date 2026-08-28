# NeMo AutoModel を Amazon SageMaker Training Job で動かす — 構築アプローチ

対象: NVIDIA [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)（PyTorch DTensor ネイティブな LLM/VLM 学習ライブラリ）

方針:
1. AutoModel 導入済みのカスタムコンテナを作成し、SageMaker Training Job で使用する
2. 学習スクリプト `train.py` と、Jupyter Notebook ベースのジョブ実行スクリプトを用意する

本ドキュメントは実装前の**構築アプローチ（手順）の検討結果**です。

## 0. 確定スコープ

ヒアリングの結果、以下を前提として設計を確定しました。

| 項目 | 決定 | 設計への影響 |
|---|---|---|
| 学習規模 | **7B〜13B 級 / 単一ノード多 GPU**（`ml.p4d.24xlarge` = 8×A100 等） | **EFA / aws-ofi-nccl 対応が不要**になり、コンテナが大幅に単純化される |
| 学習の種類 | **LoRA / PEFT** | オプティマイザ状態と保存対象が小さく、VRAM・チェックポイント容量ともに余裕が出る |
| データ経路 | **S3 チャネル経由** | `dataset.path_or_dataset` を `SM_CHANNEL_TRAIN` に写像する。ネットワーク分離も選択可能 |

この結果、**当面マルチノードは対象外**です。将来的な拡張余地は残しますが（`distribution` に
`torch_distributed` を使う設計はそのまま多ノードへ拡張できます）、Phase 6 の EFA 検証は
スコープ外とします。

---

## 1. 事前調査で確定した事実

実装方針の前提になるため、AutoModel 本体のソースを読んで確認した内容を先に整理します。

### 1.1 AutoModel の起動方式

`nemo_automodel/cli/app.py` の docstring に、**外部 torchrun からの起動が公式にサポートされている**と明記されています。

```
# Recommended — the CLI handles torchrun internally:
automodel <config.yaml> [--nproc-per-node N] [--key.subkey=override ...]

# Also supported — external torchrun launch:
torchrun --nproc-per-node N -m nemo_automodel.cli.app <config.yaml> [--key.subkey=override ...]
```

さらに `nemo_automodel/components/launcher/interactive.py` の `InteractiveLauncher` は次のように動作します。

```python
@staticmethod
def _is_torchrun_worker() -> bool:
    # torchrun は全ワーカーに LOCAL_RANK と TORCHELASTIC_RUN_ID の両方を設定する
    return "LOCAL_RANK" in os.environ and "TORCHELASTIC_RUN_ID" in os.environ
```

torchrun 配下と判定されると `torchrun` を再起動せず、その場（in-process）でレシピを実行します。

```python
recipe_cls = resolve_recipe_cls(recipe_target)
recipe = recipe_cls(config)
recipe.setup()
return recipe.run_train_validation_loop()
```

**これが本構築の要**です。SageMaker の `distribution={"torch_distributed": {"enabled": True}}` は
まさに `torchrun` でユーザースクリプトを起動する仕組みなので、
「SageMaker が torchrun を張る → その配下で AutoModel が in-process 実行される」
という形に **二重起動なしで**きれいに嵌ります。SLURM 版 `slurm.sub` と同じ構図です。

### 1.2 設定ファイル（YAML）の仕組み

- YAML の `recipe:` キーで実行するレシピクラスを指定する
  （例: `recipe: TrainFinetuneRecipeForNextTokenPrediction`）
- CLI から `--dotted.key=value` 形式で**任意のフィールドを上書きできる**
  （`nemo_automodel/components/config/_arg_parser.py` の `parse_args_and_load_config`）
- `_target_` による Hydra 風のオブジェクト指定が全面的に使われている

主要セクションの例（`examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag_peft.yaml` より）:

```yaml
recipe: TrainFinetuneRecipeForNextTokenPrediction
step_scheduler:   { global_batch_size: 64, local_batch_size: 8, ckpt_every_steps: 1000, num_epochs: 1 }
model:            { _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained,
                    pretrained_model_name_or_path: meta-llama/Llama-3.2-1B }
checkpoint:       { enabled: true, checkpoint_dir: checkpoints/,
                    model_save_format: safetensors, save_consolidated: true }
peft:             { _target_: nemo_automodel.components._peft.lora.PeftConfig,
                    target_modules: '*_proj', dim: 8, alpha: 32 }
distributed:      { strategy: fsdp2, dp_size: none, tp_size: 1, cp_size: 1 }
dataset:          { _target_: ...datasets.llm.hellaswag.HellaSwag, path_or_dataset: rowan/hellaswag, split: train }
optimizer:        { _target_: torch.optim.Adam, lr: 1.0e-5 }
```

**この「YAML + ドット記法上書き」が、SageMaker の hyperparameters と 1:1 で対応します。**
`train.py` は基本的にこの変換を行うだけの薄いアダプタで済みます。

### 1.3 パッケージ / コンテナ

| 項目 | 内容 |
|---|---|
| PyPI | `nemo-automodel`（`requires-python >=3.10`, `torch>=2.6.0`） |
| CLI エントリポイント | `automodel` / `am` → `nemo_automodel.cli.app:main` |
| 公式コンテナ | `nvcr.io/nvidia/nemo-automodel:25.11`（NGC カタログ） |
| ベース | `nvcr.io/nvidia/cuda-dl-base` または `nvcr.io/nvidia/pytorch`（`docker/Dockerfile` の ARG） |
| venv | `/opt/venv`（`PATH` に追加済み） |
| WORKDIR | `/opt/Automodel` |
| 実行ユーザ | `USER 65532:65532`（非 root） |
| 同梱 | TransformerEngine, FlashAttention 3, DeepEP, bitsandbytes, torchao などビルド済み |

TransformerEngine や FlashAttention 3 は**ソースビルドに数十分〜数時間**かかるため、
これらがビルド済みである公式コンテナを土台にする価値が非常に高い、という点が
後述のコンテナ方針の判断根拠になります。

---

## 2. 全体アーキテクチャ

```
[ Notebook (SageMaker Studio / ローカル) ]
        │  sagemaker.pytorch.PyTorch(image_uri=<自前ECR>, entry_point="train.py",
        │                            distribution={"torch_distributed": {"enabled": True}})
        │  .fit({"train": s3://...})
        ▼
[ SageMaker Training Job ]
        │  ECR からカスタムコンテナを pull
        │  S3 の入力データを /opt/ml/input/data/<channel> に配置
        │  source_dir を /opt/ml/code に展開
        ▼
[ sagemaker-training toolkit ]
        │  torchrun --nnodes=N --nproc_per_node=<GPU数>
        │           --master_addr=<algo-1> --node_rank=<i> train.py --<hp>=<value>
        ▼
[ train.py （各 rank で 1 プロセス） ]
        │  SageMaker の環境変数/パス → AutoModel の YAML 上書き引数に変換
        │  sys.argv を組み立てて nemo_automodel.cli.app.main() を呼ぶ
        ▼
[ AutoModel InteractiveLauncher ]
        │  LOCAL_RANK + TORCHELASTIC_RUN_ID を検出 → in-process 実行（torchrun 再起動なし）
        ▼
[ TrainFinetuneRecipeForNextTokenPrediction ]
           FSDP2 / TP / CP で学習
           checkpoint_dir → /opt/ml/checkpoints（S3 と自動同期）
           最終成果物     → /opt/ml/model（model.tar.gz として S3 へ）
```

---

## 3. コンテナ方針の比較と選定

### 案 A（推奨）: NGC の AutoModel 公式イメージをベースに、SageMaker 対応を足す

```dockerfile
FROM nvcr.io/nvidia/nemo-automodel:25.11
USER root
# SageMaker script mode を有効にする
RUN /opt/venv/bin/pip install --no-cache-dir sagemaker-training
# EFA / aws-ofi-nccl（マルチノード時に必須。5.4 参照）
...
```

- ✅ TE / FA3 / DeepEP / bitsandbytes がビルド済み → イメージ作成が現実的な時間で終わる
- ✅ NVIDIA が検証した依存関係の組み合わせをそのまま使える
- ⚠️ EFA 用の `aws-ofi-nccl` プラグインが入っていないため自前で追加が必要
- ⚠️ イメージが巨大（数十 GB）→ ビルド環境とジョブ起動時間に影響

### 案 B: AWS Deep Learning Container を拡張する

```dockerfile
FROM 763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:<ver>-gpu-py311-cu124-ubuntu22.04-sagemaker
RUN pip install nemo-automodel[all]
```

- ✅ sagemaker-training toolkit / EFA / aws-ofi-nccl が最初から入っている
- ❌ DLC の torch/CUDA バージョンと AutoModel の要求（`torch>=2.6`, cu129/cu130 系 index）が衝突しやすい
- ❌ TransformerEngine / FlashAttention 3 をソースビルドすることになり、ビルド時間と失敗リスクが大きい

### 結論

**案 A を採用**します。AutoModel は依存が非常に重く、NVIDIA がビルド済みの成果物を捨てる
コストが割に合いません。案 A で不足するのは SageMaker 連携部（toolkit と EFA）だけであり、
そこは追加が容易です。

なお、まず単一ノードで動かすだけなら EFA 対応は後回しにでき、案 A の追加作業は
`pip install sagemaker-training` の 1 行だけで済みます。**段階的に進められる**点も案 A の利点です。

---

## 4. リポジトリ構成（予定）

```
.
├── README.md
├── docs/
│   └── 00_approach.md              # 本ドキュメント
├── container/
│   ├── Dockerfile                  # 案 A ベース
│   ├── build_and_push.sh           # NGC pull → build → ECR push
│   └── requirements-sagemaker.txt
├── src/                            # source_dir としてジョブごとにアップロードされる
│   ├── train.py                    # SageMaker ↔ AutoModel アダプタ（エントリポイント）
│   ├── sm_paths.py                 # SageMaker の規約パス/環境変数の解決
│   └── requirements.txt            # ジョブ実行時に追加インストールしたいもの（任意）
├── configs/
│   ├── llama3_2_1b_squad_sm.yaml   # SageMaker 用ベース設定
│   └── llama3_2_1b_squad_peft_sm.yaml
└── notebooks/
    └── 01_launch_training_job.ipynb
```

`train.py` と `configs/` を `source_dir` に含めることで、**設定やスクリプトの変更で
コンテナを再ビルドする必要がなくなります**（再ビルドが必要なのは依存パッケージを
変えるときだけ）。これは巨大イメージを扱ううえで重要な設計判断です。

---

## 5. 各コンポーネントの設計

### 5.1 `train.py`（薄いアダプタ）

役割は「SageMaker の規約 → AutoModel の設定上書き」への変換のみ。学習ループは書きません。

```python
# 概略（実装イメージ）
import os, sys, json, pathlib

def main():
    # 1) SageMaker が渡した hyperparameters を argparse で受ける
    #    --config           : configs/ 配下のベース YAML 名
    #    --model_id, --lr, --num_epochs ... : よく使うものは専用フラグ
    #    それ以外は --set "a.b.c=value" で素通し
    #
    # 2) SageMaker のパス/環境変数を AutoModel の設定に写像（下表）
    overrides = [
        f"--checkpoint.checkpoint_dir={os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')}",
        f"--dataset.path_or_dataset={os.environ['SM_CHANNEL_TRAIN']}",
        ...
    ]
    #
    # 3) argv を組み立てて AutoModel CLI をそのまま呼ぶ
    #    torchrun 配下なので InteractiveLauncher が in-process 実行してくれる
    sys.argv = ["automodel", str(config_path), *overrides, *passthrough]
    from nemo_automodel.cli.app import main as automodel_main
    rc = automodel_main()
    #
    # 4) rank 0 のみ、最終成果物を /opt/ml/model へ配置
    ...
```

`nemo_automodel.cli.app.main()` に委譲することで、AutoModel 側のレシピ追加・
ランチャー変更に自動的に追従できます（`recipe:` キーを見て適切なクラスを解決するため、
LLM SFT / PEFT / VLM / KD などをすべて同じ `train.py` で扱えます）。

### 5.2 パスと設定のマッピング

| SageMaker 側 | 環境変数 | AutoModel 設定 |
|---|---|---|
| 学習データ `/opt/ml/input/data/train` | `SM_CHANNEL_TRAIN` | `dataset.path_or_dataset` |
| 検証データ `/opt/ml/input/data/validation` | `SM_CHANNEL_VALIDATION` | `validation_dataset.path_or_dataset` |
| 事前学習重み `/opt/ml/input/data/model` | `SM_CHANNEL_MODEL` | `model.pretrained_model_name_or_path` |
| チェックポイント `/opt/ml/checkpoints` | — | `checkpoint.checkpoint_dir` |
| 最終成果物 `/opt/ml/model` | `SM_MODEL_DIR` | rank 0 で consolidated 重みをコピー |
| ログ/メトリクス `/opt/ml/output/data` | `SM_OUTPUT_DATA_DIR` | JSONL ログの出力先 |
| GPU 数 | `SM_NUM_GPUS` | `--nproc_per_node`（toolkit が設定） |
| ノード一覧 / 自ホスト | `SM_HOSTS`, `SM_CURRENT_HOST` | `MASTER_ADDR` / `NODE_RANK`（toolkit が設定） |
| グローバルバッチ | hyperparameter | `step_scheduler.global_batch_size` |

`/opt/ml/checkpoints` は `checkpoint_s3_uri` を指定すると**ジョブ開始時に S3 から
ダウンロードされ、学習中に S3 へ継続同期**されます。これにより Spot 中断からの再開と
AutoModel の `ckpt_every_steps` が素直に噛み合います。

### 5.3 Notebook（`01_launch_training_job.ipynb`）

構成:

1. セットアップ（`role`, `region`, `bucket`, ECR イメージ URI の確認）
2. データ準備（HF データセットを S3 へ配置、または HF Hub から直接読む構成）
3. `PyTorch` Estimator の定義

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    image_uri=f"{account}.dkr.ecr.{region}.amazonaws.com/nemo-automodel-sagemaker:25.11",
    entry_point="train.py",
    source_dir="../src",          # configs もここに含める
    role=role,
    instance_type="ml.g5.12xlarge",   # 検証用。本番は p4d/p5
    instance_count=1,
    distribution={"torch_distributed": {"enabled": True}},
    hyperparameters={
        "config": "llama3_2_1b_squad_peft_sm.yaml",
        "set": "step_scheduler.num_epochs=1,optimizer.lr=1e-4,peft.dim=16",
    },
    environment={"HF_TOKEN": ..., "HF_HOME": "/tmp/hf"},
    checkpoint_s3_uri=f"s3://{bucket}/automodel/ckpt/",
    keep_alive_period_in_seconds=1800,   # ウォームプールで反復を高速化
    max_run=24*3600,
)
estimator.fit({"train": f"s3://{bucket}/data/train"})
```

4. ジョブ監視（CloudWatch Logs / `TrainingJobAnalytics`）
5. 成果物の取得と簡単な推論確認
6. （付録）マルチノード設定、Spot 設定、`instance_type="local_gpu"` でのローカルモード

---

## 6. 構築手順（フェーズ分割）

各フェーズに完了条件を置き、前フェーズが通ってから次に進む形にします。

### Phase 0: 前提整備
- AWS: S3 バケット、ECR リポジトリ、SageMaker 実行ロール（S3/ECR/CloudWatch 権限）
- **サービスクォータの引き上げ申請**（`ml.g5.12xlarge` / `ml.p4d.24xlarge` などの
  training job usage）。承認に数日かかることがあるため**最初に着手**する
- NGC アカウントと API キー（`nvcr.io` からの pull に必要）
- HuggingFace トークン（gated モデルを使う場合）
- **完了条件**: 目的のインスタンスタイプのクォータが 1 以上ある

### Phase 1: AutoModel 単体の動作確認（SageMaker 抜き）
- GPU 付き EC2 で `nvcr.io/nvidia/nemo-automodel:25.11` を起動
- `automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node 1` を実行
- 続いて **外部 torchrun 経路**も確認:
  `torchrun --nproc-per-node 2 -m nemo_automodel.cli.app <config> --step_scheduler.num_epochs=1`
- **完了条件**: 外部 torchrun 経路で数ステップ学習が回り、in-process 実行のログが出る
- ここで詰まる要因（モデルのライセンス同意、VRAM 不足、データセット取得）を
  SageMaker より先に潰しておくのが目的

### Phase 2: SageMaker 用コンテナの作成
- `container/Dockerfile`（案 A）を作成: `USER root` → `pip install sagemaker-training`
- ビルド環境の選択（イメージが巨大なため重要）:
  - SageMaker Studio から `sm-docker build`（CodeBuild 実行、ローカル docker 不要）
  - または EBS を大きめ（200 GB 以上）にした EC2 で `docker build`
- ECR へ push（**Training Job と同一リージョン**であること）
- **完了条件**: ECR のイメージを `docker run` して `automodel --help` が通る

### Phase 3: `train.py` の実装
- 5.1 / 5.2 の設計に沿って実装
- ローカル（Phase 1 の EC2）で SageMaker のディレクトリ構造を手で作って擬似実行し、
  パス写像が正しいことを確認
- **完了条件**: `/opt/ml/...` 相当のパスを使って torchrun 経由で学習が回る

### Phase 4: 設定ファイルの整備
- `configs/` に SageMaker 前提のベース YAML を用意
  （`checkpoint_dir` や `dataset` はどうせ `train.py` が上書きするので既定値でよい）
- まず 1B クラスの小さいモデル（`meta-llama/Llama-3.2-1B`）で通し、
  その後ターゲットモデルへ差し替える

### Phase 5: Notebook の実装と単一ノード実行
- 5.3 の構成で Notebook を作成
- `instance_count=1` で実行 → CloudWatch にログ、S3 に成果物が出ることを確認
- **完了条件**: `model.tar.gz` に consolidated な safetensors が入っている

### Phase 6: スケールと運用性の確認（単一ノード）
1. 1 GPU → 1 ノード多 GPU（`ml.p4d.24xlarge` = 8×A100 40GB / `ml.g5.48xlarge` = 8×A10G 24GB）
   - 13B 級の LoRA は FSDP2 でシャードすれば `ml.g5.48xlarge` でも収まる見込みだが、
     余裕を見るなら `ml.p4d.24xlarge` を第一候補とする
   - `distributed.tp_size` / `cp_size` は 1 のままで、まず FSDP2 のみで通す
2. チェックポイントからの再開（`checkpoint_s3_uri` を再利用して 2 回目のジョブを実行）
3. Spot 利用（`use_spot_instances=True`, `max_wait`）での中断・再開
4. LoRA アダプタのみのマージ／推論確認
- **完了条件**: 8 GPU でのスループットが 1 GPU の 6 倍以上、かつ中断したジョブが
  チェックポイントから正しく再開できる

## 7. 想定される落とし穴と対策

| # | 論点 | 内容と対策 |
|---|---|---|
| 1 | **EFA / aws-ofi-nccl** | （**今回はスコープ外**）NGC イメージには AWS の `aws-ofi-nccl` プラグインが入っておらず、無いとマルチノードの NCCL が EFA を使えず TCP にフォールバックして著しく遅くなる。単一ノード構成では不要。将来マルチノードへ拡張する際は Dockerfile に EFA installer + `aws-ofi-nccl` の追加が必要になる |
| 2 | **イメージサイズ** | 数十 GB。ECR pull がジョブ起動時間に直撃する（初回 10 分超もあり得る）。`keep_alive_period_in_seconds`（ウォームプール）で反復開発時の再 pull を回避する |
| 3 | **非 root ユーザ** | 公式イメージは `USER 65532`。SageMaker は `/opt/ml` 配下に root 所有で書き込むため、`USER root` に戻すのが無難 |
| 4 | **ENTRYPOINT** | script mode では `train` コマンド（toolkit が提供）が実行される。ベースイメージの ENTRYPOINT がそれを妨げないか確認し、必要なら `ENTRYPOINT []` でクリアする |
| 5 | **`/dev/shm`** | DataLoader の worker が shm 不足で落ちることがある。`dataloader.num_workers` を控えめにするか、shm 使用量を抑える設定にする |
| 6 | **ネットワーク分離** | `enable_network_isolation=True` だと HF Hub にアクセスできない。その場合はモデル重みとデータセットを事前に S3 へ置き、`model` チャネルとして渡す |
| 7 | **HF_TOKEN の扱い** | Notebook にベタ書きしない。SageMaker の `environment` に渡すか、Secrets Manager から取得する。`HF_HOME` は `/tmp` 配下など書き込み可能な場所に向ける |
| 8 | **グローバルバッチの整合** | `global_batch_size` は `local_batch_size × データ並列数` で割り切れる必要がある。`instance_count` × `SM_NUM_GPUS` から `train.py` 側で検証・自動調整するとジョブ失敗を減らせる |
| 9 | **`max_run` の上限** | Training Job の既定は 24 時間、上限は 28 日。長時間学習ではチェックポイント再開を前提に設計する |
| 10 | **クォータ** | Phase 0 に書いた通り、GPU インスタンスのクォータ承認が最大のリードタイム要因になりやすい |

---

## 8. 次のアクション

スコープが確定したため、Phase 0 と Phase 2〜5 を並行して進められます。

1. **Phase 0（先行着手）** — `ml.p4d.24xlarge`（または `ml.g5.48xlarge`）の
   training job クォータ引き上げ申請。承認待ちが最大のリードタイム要因になるため最優先。
   あわせて NGC API キーと HuggingFace トークンを準備する
2. **Phase 2** — `container/Dockerfile` の作成と ECR への push
   （EFA 対応が不要になったため、実質 `USER root` + `pip install sagemaker-training` のみ）
3. **Phase 3〜4** — `train.py` と PEFT 用のベース設定 YAML の実装
4. **Phase 5** — Notebook を作成し、`ml.g5.12xlarge` あたりで小さいモデルから
   エンドツーエンドを通す

最初のマイルストーンは「**1B モデル + LoRA + 単一ノードで S3 チャネルからデータを読み、
S3 にアダプタが出力される**」ところまでを通すこととします。ここが通れば、
ターゲットモデルへの差し替えは設定変更で済みます。
