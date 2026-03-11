# Augment_tableQA
This is the implementation for the paper: [Augment before You Try: Knowledge-Enhanced Table Question Answering via Table Expansion](https://arxiv.org/abs/2401.15555).

### Dependencies


```
pip install requirements.txt
```

### Data download

Already included in repo

### Evaluation

(Loading the model and running each baseline is provided in each associated script)

- **AugmentQA** - run files as per the dataset you want to evaluate on `run_augment_{dataset_name}.py` using VLLM config present in the `VLLM_SETUP.md` in `Augment_QA/endtoend` branch
- **End to End** - run files as per the dataset you want to evaluate on `run_endtoend_{dataset_name}.py` using VLLM config present in the `VLLM_SETUP.md` in `Augment_QA/endtoend` branch
- **PoT and Binder baselines** - run all cells in `run.ipynb`
- **CoT** - run all cells in `table_cot_experiments.ipynb`

