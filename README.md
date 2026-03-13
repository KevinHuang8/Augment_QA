# Augment_tableQA
This is the implementation for the paper: [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555).

### Dependencies

```bash
pip install -r requirements.txt
```

For End-to-End and AugmentQA, vLLM must also be installed and running to serve the model locally. See the `VLLM_SETUP.md` in the `endtoend` branch for full setup instructions. Requires approximately 12 GB VRAM. For CoT, PoT, and Binder baselines, running the notebooks mentioned in the evaluation section handles it. The commands such as evaluation and scoring commands are provided in the notebooks as well.

### Pretrained Model

We use [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct).
### Data Download

All datasets are either already included in the repo or downloaded at runtime. No additional download is required.

### Preprocessing

No preprocessing is required beyond cloning the repository. The datasets are pre-filtered to match the experimental setup described in the paper.

### Evaluation

(Loading the model and running each baseline is provided in each associated script)

- **AugmentQA** - run files as per the dataset you want to evaluate on `run_augment_{dataset_name}.py` using VLLM config present in the `VLLM_SETUP.md` in `Augment_QA/endtoend` branch
- **End to End** - run files as per the dataset you want to evaluate on `run_endtoend_{dataset_name}.py` using VLLM config present in the `VLLM_SETUP.md` in `Augment_QA/endtoend` branch
- **PoT and Binder baselines** - run all cells in `run.ipynb`
- **CoT** - run all cells in `table_cot_experiments.ipynb`

