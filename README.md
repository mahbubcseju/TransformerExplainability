# TransformerExplainability
A explainability tools for Transformer-based deep learning model for Vulnerability Detection



## Requirements

> python3.7

All the python package requirements could be found in `requirements.txt` file.

## Dataset

We published our dataset in the following link. After downloading the dataset, put it in the   `data/` directory.

[Download Link](https://iastate.box.com/s/whfryutr2e4thu3qxkhhmci7hyfnduvi)


## Training CodeBERT model
Before running any script, set the corresponding paths in   `code/config.py`.

Run the following command:

```
    python code/train_and_test.py --dataset func_jsonl --batch 128
```

## Running our approach

At first, run the following command:

```
    python explain_examples.py
```

Then, run the following notebook:

```
calculate_line_score.ipynb
```
