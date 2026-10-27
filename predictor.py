import pickle
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = Path(__file__).resolve().parent / "model"


class NextWordPredictor:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        self.model = load_model(model_dir / "lstm_model.h5", compile=False)
        with open(model_dir / "tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(model_dir / "max_len.pkl", "rb") as f:
            self.max_len = pickle.load(f)
        self.index_word = {i: w for w, i in self.tokenizer.word_index.items()}

    def predict(self, text, top_k=5):
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        if not sequence:
            return []
        padded = pad_sequences(
            [sequence[-self.max_len:]], maxlen=self.max_len, padding="pre"
        )
        probs = self.model.predict(padded, verbose=0)[0]
        ranked = np.argsort(probs)[::-1]
        results = []
        for idx in ranked:
            word = self.index_word.get(int(idx))
            if word is None:
                continue
            results.append((word, float(probs[idx])))
            if len(results) == top_k:
                break
        return results
