# STSScore

This repository accompanies the paper ["Semantic similarity prediction is better than other semantic similarity measures"](https://openreview.net/forum?id=bfsNmgN5je) accepted for publication at the [Transactions of Machine Learning Research (TMLR)](https://jmlr.org/tmlr/). 

# Overview

The repository contains a Jupyter notebook that can be used to reproduce our results, the computed similarity results, and the plots from the paper. We do not provide a Python package for your use. If you want to use STSScore in your own work, all you need is to install the [Huggingface transformers library](https://huggingface.co/docs/transformers/installation) and add the following code to your project:

```python
import transformers

class STSScorer:
    def __init__(self):
        model_name = 'WillHeld/roberta-base-stsb'
        self._sts_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._sts_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name)
        self._sts_model.eval()

    def score(self, sentence1, sentence2):
        sts_tokenizer_output = self._sts_tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt")
        sts_model_output = self._sts_model(**sts_tokenizer_output)
        return sts_model_output['logits'].item()/5
```

You can then compute the semantic similarity between two sentences as follows:

```python
scorer = STSScorer()
score = scorer.score("I like apples", "I like oranges")
```

Enjoy!

# Automatically move to GPU if available

If you want to automatically move the model to the GPU if available, you can use the following code:

```python
import transformers
import torch

class STSScorer:
    def __init__(self):
        model_name = 'WillHeld/roberta-base-stsb'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._sts_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._sts_model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self._sts_model.eval()

    def score(self, sentence1, sentence2):
        sts_tokenizer_output = self._sts_tokenizer(sentence1, sentence2, padding=True, truncation=True, return_tensors="pt").to(self.device)
        sts_model_output = self._sts_model(**sts_tokenizer_output)
        return sts_model_output['logits'].item()/5
```

# Citation

If you use STSScore in your work, please cite our paper:

```
@article{herbold2023semantic,
      title={Semantic similarity prediction is better than other semantic similarity measures}, 
      author={Steffen Herbold},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2024},
      url={https://openreview.net/forum?id=bfsNmgN5je},
}
```
