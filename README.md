# Augment_tableQA
This is the implementation for the paper: [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555).

## Requirements
### Environment
Install conda environment by running
```bash
conda env create -f environment.yml
conda activate augment
```

## Setup
### 1) vLLM endpoint (open-source models)
Inference uses a local vLLM server. Configure it once:

1. Copy `vllm_config.example.json` to `vllm_config.json` and set `base_url` and `model`:
   ```bash
   cp vllm_config.example.json vllm_config.json
   # Edit vllm_config.json — set "base_url" and "model"
   ```

2. Create `key.txt` in the project root (not used for auth, but required by scripts):
   ```bash
   echo "dummy" > key.txt
   ```

3. Start the vLLM server (see [VLLM_SETUP.md](VLLM_SETUP.md) for full details):
   ```bash
   bash start_vllm_server.sh
   ```

4. Verify the server is running:
   ```bash
   python test_vllm_model.py --list-models
   ```

## Running Experiments

All commands below should be run from the `Augment_QA/` directory:
```bash
cd Augment_QA
```

The model name is automatically read from `vllm_config.json` by each runscript.

### AugmentQA (main method)

Runs a two-step pipeline: (1) generate augmentation programs, (2) execute programs + evaluate.

```bash
# FinQA (test split)
python runscripts/run_augment_finqa.py

# TatQA (validation split)
python runscripts/run_augment_tatqa.py

# WikiTableQuestions / missing_squall (validation split)
python runscripts/run_augment_wikitq.py
```

### End-to-End baseline

Direct QA without table augmentation or SQL generation.

```bash
# FinQA (test split)
python runscripts/run_endtoend_finqa.py

# TatQA (validation split)
python runscripts/run_endtoend_tatqa.py

# WikiTableQuestions / missing_squall (validation split)
python runscripts/run_endtoend_wikitq.py
```

### PoT (Program of Thought) baselines

```bash
# WikiTQ (test split)
python runscripts/run_pot_wikitq.py

# FinQA (test split)
python scripts/execute_pot_finqa.py \
  --dataset finqa --dataset_split test \
  --api_config_file key.txt \
  --output_program_file pot_finqa_test.json

# TatQA (validation split)
python scripts/execute_pot_tatqa.py \
  --dataset tatqa --dataset_split validation \
  --api_config_file key.txt \
  --output_program_file pot_tatqa_validation.json
```

### Binder CoT baseline

```bash
# WikiTQ / missing_squall (validation split)
python runscripts/run_bindercot_wikitq.py
```

Output is stored in `results/` and performance metrics are printed to stdout.

**Note:** There may be ~1% random performance variation even with greedy decoding. Re-run if results don't match the paper exactly.

## References
If you found our reproduction useful or liked this work, please consider citing the original paper:
```
@misc{liu2024augment,
      title={Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion}, 
      author={Yujian Liu and Jiabao Ji and Tong Yu and Ryan Rossi and Sungchul Kim and Handong Zhao and Ritwik Sinha and Yang Zhang and Shiyu Chang},
      year={2024},
      eprint={2401.15555},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Our implementation is based on the following repos:
* https://github.com/xlang-ai/Binder
* https://github.com/wenhuchen/Program-of-Thoughts
