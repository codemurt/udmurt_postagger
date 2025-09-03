# Udmurt POS-Tagger

Python library for part-of-speech (POS) tagging of Udmurt language sentences using a bidirectional LSTM neural network model.

## Features

- Trained on the [udmurtNLP/zerpal-pos-tagging](https://huggingface.co/datasets/udmurtNLP/zerpal-pos-tagging) dataset
- Supports 17 POS tags
- Achieves 91.83% accuracy on the test set
- Handles out-of-vocabulary words using special embeddings

## Installation

```bash
git clone https://github.com/codemurt/udmurt_postagger.git
cd udmurt_postagger
pip install .
```

## Usage

```py
from udmurt_postagger import UdmurtPOSTagger

tagger = UdmurtPOSTagger()
sentence = "Мон мынӥсько школае .".split()
tags = tagger.predict(sentence)

print("Token -> POS Tag")
for token, tag in zip(sentence, tags):
    print(f"{token} -> {tag}")
```

## Model Architecture

The model uses a bidirectional LSTM architecture with the following layers:
- **Embedding Layer**: 128-dimensional word embeddings
- **Bidirectional LSTM**: 256 units (512 total output dimensions)
- **TimeDistributed Dense Layer**: 18 units (17 tags + padding)
- **Activation**: Softmax activation for tag probability distribution

**Total Parameters**: 3,060,754 (11.68 MB)

## Performance

Trained on 9,913 sentences and evaluated on 2,479 sentences:

| Tag      | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| N        | 0.8740    | 0.9058 | 0.8896   | 5957    |
| V        | 0.8994    | 0.9062 | 0.9028   | 6261    |
| PRO      | 0.9889    | 0.9797 | 0.9843   | 1724    |
| CNJ      | 0.7901    | 0.8081 | 0.7990   | 787     |
| PART     | 0.9162    | 0.8961 | 0.9061   | 1916    |
| NUM      | 0.9787    | 0.9200 | 0.9485   | 250     |
| IMIT     | 0.7778    | 0.4038 | 0.5316   | 104     |
| ADV      | 0.8628    | 0.8332 | 0.8477   | 1283    |
| O        | 0.9998    | 0.9956 | 0.9977   | 6199    |
| INTRJ    | 0.9500    | 0.8636 | 0.9048   | 66      |
| ADJPRO   | 0.9769    | 0.9270 | 0.9513   | 137     |
| ADJ      | 0.8024    | 0.7357 | 0.7676   | 927     |
| PREDIC   | 0.9200    | 0.9718 | 0.9452   | 71      |
| POST     | 0.9343    | 0.9309 | 0.9326   | 275     |
| ADVPRO   | 0.9972    | 0.9757 | 0.9863   | 370     |
| PREP     | 0.5000    | 0.7500 | 0.6000   | 4       |
| PARENTH  | 0.8889    | 0.9275 | 0.9078   | 69      |

**Overall Accuracy**: 91.83%  
**Weighted F1-Score**: 91.79%

## Training Details

- **Optimizer**: Adam (learning rate 0.001)
- **Batch Size**: 128
- **Epochs**: 40
- **Padding**: Post-padding to maximum sequence length (89 tokens)
- **Special Tokens**: 
  - `-PAD-` for padding
  - `-OOV-` for out-of-vocabulary words

## File Descriptions

- `udmurt_pos_tagger_model.h5`: Trained Keras model
- `udmurt_pos_word2index.pkl`: Word-to-index mapping dictionary
- `udmurt_pos_index2tag.pkl`: Index-to-tag mapping dictionary