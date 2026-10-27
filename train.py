import argparse
import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_SIZE = 10000
EMBEDDING_DIM = 50
LSTM_UNITS = 128


def build_sequences(lines, tokenizer, max_seq_len):
    inputs, targets = [], []
    for tokens in tokenizer.texts_to_sequences(lines):
        tokens = tokens[:max_seq_len]
        for i in range(1, len(tokens)):
            inputs.append(tokens[:i])
            targets.append(tokens[i])
    max_len = max(len(s) for s in inputs)
    X = pad_sequences(inputs, maxlen=max_len, padding="pre")
    y = np.array(targets)
    return X, y, max_len


def main():
    parser = argparse.ArgumentParser(description="Train the LSTM next word model.")
    parser.add_argument("--data", required=True, help="Text file, one sentence/line per row")
    parser.add_argument("--out-dir", default="model", help="Where to save model artifacts")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=50, help="Max tokens used per line")
    args = parser.parse_args()

    lines = [l.strip() for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise SystemExit("The data file is empty.")

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(lines)

    X, y, max_len = build_sequences(lines, tokenizer, args.max_seq_len)
    print(f"Lines: {len(lines)} | Samples: {len(X)} | Vocab: {len(tokenizer.word_index)} | Input length: {max_len}")

    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        LSTM(LSTM_UNITS),
        Dense(VOCAB_SIZE, activation="softmax"),
    ])
    model.build((None, max_len))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "lstm_model.h5")
    with open(out_dir / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    with open(out_dir / "max_len.pkl", "wb") as f:
        pickle.dump(max_len, f)
    print(f"Saved model, tokenizer and max_len to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
