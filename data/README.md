# 学習データ

## cooking_basics — 料理の基礎知識（SFT 用サンプル）

Qwen3.5 での LoRA SFT 疎通確認用に作成した、料理の基礎知識に関する日本語の instruction データです。
`data/build_cooking_dataset.py` に全件をハードコードしており、実行すると `cooking_basics/` を再生成します。

| ファイル | 件数 | 用途 |
|---|---|---|
| `all.json` / `all.jsonl` | 306 | 全件 |
| `train.json` / `train.jsonl` | 245 | 学習用（8 割） |
| `val.json` / `val.jsonl` | 61 | 検証用（2 割） |

分割は `seed=42` で固定しています。`.json` は配列形式、`.jsonl` は 1 行 1 レコードで、内容は同一です。

### カテゴリ

切り方・下ごしらえ / 加熱調理法 / 調味料と味付け / 計量・換算 / だし・うま味 / 米・麺・パン・粉 /
肉 / 魚介 / 卵・乳製品 / 野菜・果物 / 保存・解凍 / 衛生・食の安全 / 道具・器具 / 失敗の原因と対処
の 14 カテゴリ、各 15〜30 件です。

### レコード形式

```json
{
  "instruction": "次の食材に適した下ごしらえを答えてください。",
  "input": "こんにゃく",
  "output": "こんにゃくは下ゆでをして臭みを抜くのが基本の下ごしらえです。…",
  "prompt": "次の食材に適した下ごしらえを答えてください。\n\nこんにゃく"
}
```

- `instruction` / `input` / `output` — 既存 Composer 版スクリプトと同じ Alpaca 形式。`input` は約 12% のレコードで非空
- `prompt` — `instruction + "\n\n" + input`（`input` が空なら `instruction` のみ）を結合したもの。
  既存 Composer 版の `generate_instruction_prompt` と同じ結合ルールです
- `output` は必ず文字列（list を含まないため、既存スクリプトにあった型正規化は不要）
- 出力は 110〜205 文字、平均 160 文字。思考過程を含まない直接回答（Qwen の非思考モード前提）

### AutoModel での読み込み

`ColumnMappedTextInstructionDataset` は `context` を **system ロール**、`question` を user ロールに入れます。
`input` を system に入れるのは不自然なので、結合済みの `prompt` を `question` に写像します。

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /opt/ml/input/data/train/train.jsonl
  column_mapping: {question: prompt, answer: output}
  use_hf_chat_template: true
  answer_only_loss_mask: true
  seq_length: 1024
```

### S3 への配置

```bash
BUCKET=<your-bucket>
aws s3 cp data/cooking_basics/train.jsonl s3://$BUCKET/data/cooking_basics/train/train.jsonl
aws s3 cp data/cooking_basics/val.jsonl   s3://$BUCKET/data/cooking_basics/validation/val.jsonl
```

SageMaker では `train` / `validation` チャネルとして渡します（`docs/01_composer_to_automodel.md` 3.2 参照）。

### 再生成・追加

`build_cooking_dataset.py` の `add(instruction, output, input_="")` を追記して実行してください。
instruction と input の組が重複するとアサーションで止まります。
