import numpy as np
from preprocess import clean_text

with open("lyrics_raw.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text = clean_text(raw_text)
print(f"Total characters after cleaning: {len(text)}")

chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Unique characters: {vocab_size}")
print(f"Characters: {chars}")

char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for ch,i in char_to_idx.items()}

encoded_text = np.array([char_to_idx[char] for char in chars])

seq_length = 40
step = 3

sequence = []
targets = []

for i in range(0, len(encoded_text) - seq_length, step):
    seq = encoded_text[i:i+seq_length]
    target = encoded_text[i+seq_length]
    sequence.append(seq)
    targets.append(target)

print(f"Total sequences: {len(sequence)}")

X = np.array(sequence)
Y = np.array(targets)

np.save("X.npy", X)
np.save("y.npy", Y)
np.save("char_to_idx.npy", char_to_idx)
np.save("idx_to_char.npy", idx_to_char)

print("Data processed and saved")

