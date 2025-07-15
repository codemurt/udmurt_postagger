# Udmurt POS-Tagger

Python library for POS-tagging of Udmurt language sentences.

## Installation

```bash
git clone https://github.com/codemurt/udmurt-postagger.git
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


