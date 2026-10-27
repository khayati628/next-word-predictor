# 🧠 Next Word Predictor

An LSTM-based next word prediction app. Type a partial sentence and the model suggests the most likely next word, along with the top 5 candidates and their probabilities. Built with TensorFlow/Keras and a Streamlit web UI.

## Demo

```
Input:  to be or not to
Output: be (33.3%)
        remember (20.5%), make (5.7%), know, die
```

## How it works

1. Text is lowercased and converted to integer sequences with a Keras `Tokenizer` (vocabulary capped at 10,000 words).
2. The sequence is left-padded to a fixed length (`max_len`).
3. An LSTM network outputs a probability for every word in the vocabulary, and the highest ones are shown.

**Model architecture**

| Layer | Details |
|-------|---------|
| Embedding | 10,000 vocab, 50 dimensions |
| LSTM | 128 units |
| Dense | 10,000 units, softmax |

Optimizer: Adam.

## Project structure

```
next-word-predictor/
├── app.py            # Streamlit web app
├── predictor.py      # Loads the model and handles prediction
├── train.py          # Train your own model on any text file
├── requirements.txt
├── model/
│   ├── lstm_model.h5   # Trained LSTM model
│   ├── tokenizer.pkl   # Fitted tokenizer
│   └── max_len.pkl     # Input sequence length
└── README.md
```

## Getting started

```bash
git clone https://github.com/khayati628/next-word-predictor
cd next-word-predictor

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Use it from Python

```python
from predictor import NextWordPredictor

predictor = NextWordPredictor()
print(predictor.predict("to be or not to", top_k=3))
```

## Train your own model

Prepare a plain text file with one sentence (or quote, or line) per row, then run:

```bash
python train.py --data data/corpus.txt --epochs 50
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | required | Path to the text file |
| `--out-dir` | `model` | Where the model, tokenizer and `max_len` are saved |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--max-seq-len` | 50 | Max tokens used per line (lower = faster, less memory) |

The saved files are picked up automatically by the app.

## Limitations

- It predicts one word at a time from a limited vocabulary, so words outside it are ignored.
- Quality depends heavily on the training data size and training time. A small LSTM won't match modern transformer models.

## Tech stack

Python, TensorFlow / Keras, Streamlit, NumPy

## License

Add a license of your choice (MIT is a common pick).
