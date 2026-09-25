# src

SageMaker Training Job のエントリポイント `train.py` です。Estimator の `source_dir` としてジョブごとにアップロードされます。

`train.py` は、SageMaker の toolkit が `torchrun` 経由で各 rank で起動する薄いアダプタで、SageMaker の規約（入力チャネル、`/opt/ml/checkpoints`、`/opt/ml/model`、`WORLD_SIZE`）を AutoModel の YAML 設定に写像し、`nemo_automodel.cli.app.main()` に委譲します。学習ループは持ちません。

`inference/` には推論検証用のスクリプト（`infer_adapter.py`: アダプタのまま HF + PEFT で生成、`merge_adapter.py`: 本体へのマージ、`merge_adapter_mtp.py`: 本体と MTP ヘッドへのマージ）と、それらのジョブ専用の `requirements.txt` があります。学習ジョブの `source_dir` には含めず、`notebooks/02`〜`04` から `source_dir='../src/inference'` として使います。手順は `install_guide/07_inference_and_merge.md` を参照してください。

ハイパーパラメータ、規約の対応表、成果物の内容は `install_guide/05_configuration.md` を参照してください。  
ローカルでの検証方法は `install_guide/02_local_verification.md` ステップ3 を参照してください。
