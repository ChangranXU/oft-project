# OFT SFT for Code LLM + HumanEval/MBPP Evaluation

This project provides:

- OFT-based SFT training for `Qwen/Qwen2.5-3B-Instruct`
- Local dataset loading for:
  - `sahil2801/CodeAlpaca-20k`
  - `CodeResearch/Code-Evol-Instruct-OSS`
- SwanLab experiment tracking
- Code generation evaluation on:
  - `openai/openai_humaneval`
  - `Muennighoff/mbpp`
- Metrics:
  - Syntax Rate
  - Pass@1 (and optional Pass@5/10)
  - CodeBLEU

## Project Structure

- `config/`: training YAML config
- `data/dataset_info.json`: local dataset registry
- `deepspeed/`: DeepSpeed config files
- `scripts/train_sft.py`: OFT SFT trainer entrypoint
- `scripts/eval_codegen.py`: HumanEval/MBPP evaluator
- `scripts/*.sh`: convenience shell wrappers
- `utils/`: reusable config/data/OFT/SwanLab/eval helpers

## 1) Install Dependencies

```bash
cd oft-project
pip install -r requirements.txt
```

Optional acceleration:

- Install `deepspeed` if using DeepSpeed.
- Install `flash-attn` if your CUDA environment supports FlashAttention2.

## 2) Download Datasets

```bash
bash scripts/dataset_setup_code_datasets.sh
```

Downloaded files are expected at:

- `data/CodeAlpaca-20k/code_alpaca_20k.json`
- `data/Code-Evol-Instruct-OSS/code.evol.instruct.wiz.oss.json`

Dataset registry is configured in `data/dataset_info.json`.

## 3) Train with OFT SFT

Default (single GPU):

```bash
bash scripts/train_sft.sh config/qwen2_5_oft_sft.yaml
```

Multi-GPU:

```bash
NUM_GPUS=4 MASTER_PORT=29501 bash scripts/train_sft.sh config/qwen2_5_oft_sft.yaml
```

With runtime overrides:

```bash
bash scripts/train_sft.sh config/qwen2_5_oft_sft.yaml learning_rate=1e-5 num_train_epochs=2
```

With DeepSpeed (set in YAML or override):

```bash
bash scripts/train_sft.sh config/qwen2_5_oft_sft.yaml deepspeed=deepspeed/ds_z2_config.json
```

### SwanLab

`report_to: swanlab` is already enabled in the default config.

Optional config keys:

- `swanlab_project`
- `swanlab_workspace`
- `swanlab_run_name`
- `swanlab_mode` (`cloud` or `local`)
- `swanlab_api_key`
- `swanlab_logdir`

Example override:

```bash
bash scripts/train_sft.sh config/qwen2_5_oft_sft.yaml swanlab_project=oft-mini-project swanlab_run_name=qwen25-oft-sft
```

## 4) Evaluate a Model (HF ID or Local Path)

Run both benchmarks:

```bash
python scripts/eval_codegen.py --model "trained_models/qwen-3b-oft"
```

Run HumanEval only:

```bash
bash scripts/eval_humaneval.sh "trained_models/qwen-3b-oft"
```

Run MBPP only:

```bash
bash scripts/eval_mbpp.sh "trained_models/qwen-3b-oft"
```

Enable Pass@5 and Pass@10:

```bash
python scripts/eval_codegen.py --model "trained_models/qwen-3b-oft" --pass_k 1,5,10
```

Evaluation results are saved as JSON in `eval_results/`.

## 5) Metrics Definition

- **Syntax Rate**: fraction of first-sample generations that can be parsed by Python AST.
- **Pass@k**: functional correctness estimated using `evaluate` package (`code_eval`).
- **CodeBLEU**: code similarity between first prediction and reference solution.

## 6) Run Unit Tests

The project includes unit tests under `test/` for:

- data loading and preprocessing
- template/tokenization behavior
- OFT adapter target resolution and application wiring
- training helper logic

Run all tests:

```bash
cd oft-project
pytest -q test
```

Run a single test module:

```bash
pytest -q test/test_oft_adapter.py
```

## Notes

- The evaluator accepts either:
  - a base model path / HF model ID
  - an OFT adapter directory (it auto-loads base model from `adapter_config.json`)
- `qwen2_5_oft_sft.yaml` follows a LlamaFactory-like flat YAML style.
- Keep your local dataset paths consistent with `data/dataset_info.json`.
