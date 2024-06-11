# Sales Conversations Dataset Generation Using GPT-2

## Overview

This project aims to generate a comprehensive sales conversation dataset using the GPT-2 language model. The dataset is intended for research in the field of Language Models and generative AI, particularly focusing on the ability to generate coherent, contextually appropriate, and engaging conversations in a sales context. The generated conversations follow the structure of a user inquiring about a product and a salesman providing information.

## Requirements

- Python 3.6 or higher
- `transformers` library by Hugging Face
- `torch` library for PyTorch
- `csv` module (standard in Python)
- `time` and `datetime` modules (standard in Python)

## Installation

1. Install Python 3.6 or higher from [Python.org](https://www.python.org/).
2. Install the required libraries using pip:

```bash
pip install transformers torch
```

## Usage

```
run.py
```


## Script Explanation

The script `run.py` performs the following steps:

The script imports the required libraries, including:
- `transformers` for loading the GPT-2 model and tokenizer.
- Standard libraries like `csv`, `time`, and `datetime`.

Define the `generate_dialogue` Function

This function generates a response from the GPT-2 model given a prompt. It uses the following parameters to improve the diversity and relevance of the generated text:
- `no_repeat_ngram_size=2`: Prevents the model from repeating the same bigrams.
- `do_sample=True`: Enables sampling instead of greedy decoding.
- `top_k=50`: Limits the sampling pool to the top 50 words with the highest probabilities.
- `top_p=0.95`: Uses nucleus sampling to select the smallest set of words with a cumulative probability of 95%.


# Acknowledgements

This project was inspired by the research paper "Let the LLMs Talk" (2312.02913 on arXiv.org), which explores the potential of large language models to engage in meaningful and diverse conversations.

[https://arxiv.org/abs/2312.02913](https://arxiv.org/abs/2312.02913)





