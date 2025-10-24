import numpy as np
from tensorflow.keras.models import load_model

model = load_model('lyrics_model.h5')
char_to_idx = np.load("char_to_idx.npy", allow_pickle=True).item()
idx_to_char = np.load("idx_to_char.npy", allow_pickle=True).item()

vocab_size = len(char_to_idx)

def generate_text(model, seed_text, length=200, temperature=1.0):

    result = list(seed_text.lower())
    seq_length = model.input_shape[1]

    for _ in range(length):
        encoded = [char_to_idx.get(ch,0) for ch in result[-seq_length:]]
        padded = np.pad(encoded, (seq_length - len(encoded)), mode="constant")

        preds = model.predict(np.array([padded]), verbose=0)[0]

        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-8) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        next_index = np.random.choice(range(vocab_size), p=preds)
        next_char = idx_to_char[next_index]
        result.append(next_char)

    return ''.join(result)


seed_text = "hello"
print(f"Seed: {seed_text}")

generated = generate_text(model, seed_text, length=300, temperature=0.8)
print("\nGenerated Lyrics\n")
print(generated)