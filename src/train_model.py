import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# Load preprocessed data
X = np.load('X.npy', allow_pickle=True)
y = np.load('Y.npy', allow_pickle=True)
char_to_idx = np.load('char_to_idx.npy', allow_pickle=True).item()
idx_to_char = np.load('idx_to_char.npy', allow_pickle=True).item()

# Determine vocabulary size
vocab_size = len(char_to_idx)

# One-hot encode the target labels
y = to_categorical(y, num_classes=vocab_size)

# Build the LSTM model
model = Sequential([
    Embedding(vocab_size, 64, input_length=X.shape[1]),
    LSTM(128, return_sequences=False),
    Dense(vocab_size, activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Print the model summary
print(model.summary())

# Train the model
model.fit(X, y, batch_size=128, epochs=20)

# Save the trained model
model.save("lyrics_model.h5")

print("Model trained and saved as lyrics_model.h5")