# Augment_tableQA
This is the implementation for the paper: [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555).

### Dependencies

```bash
pip install -r requirements.txt
```

### Data download

Already included in repo.

### Setup

Configure vLLM before running any experiments (see [VLLM_SETUP.md](VLLM_SETUP.md) for full details):

```bash
cp vllm_config.example.json vllm_config.json   # edit base_url and model
echo "dummy" > key.txt
python test_vllm_model.py --list-models         # verify server is running
```

### Evaluation

All commands below should be run from `Augment_QA/`. Loading the model and running each baseline is provided in each associated script.

- **AugmentQA** — runs augmentation program generation + execution & evaluation:
  ```bash
  python runscripts/run_augment_finqa.py
  python runscripts/run_augment_tatqa.py
  python runscripts/run_augment_wikitq.py
  ```

- **End-to-End** — direct QA without table augmentation:
  ```bash
  python runscripts/run_endtoend_finqa.py
  python runscripts/run_endtoend_tatqa.py
  python runscripts/run_endtoend_wikitq.py
  ```

  For detailed documentation on running AugmentQA and End-to-End (setup, advanced usage, direct script invocation), see the [`endtoend`](https://github.com/KevinHuang8/Augment_QA/blob/endtoend) branch.

- **PoT and Binder baselines** — run all cells in `run.ipynb`
- **CoT** — run all cells in `table_cot_experiments.ipynb`

