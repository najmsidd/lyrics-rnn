# RNN Song Lyrics Generator (Keras/LSTM)

A learning-focused project to train a **character-level RNN (LSTM/GRU)** to generate song lyrics.

## Features
- Clean preprocessing pipeline (from multiple `.txt` files to a single corpus).
- Character-level dataset creation with configurable sequence length.
- Keras LSTM model with training callbacks (checkpointing, early stopping).
- Temperature-based text sampling for creative control.
- Clear, minimal code with comments.

## Project Structure
```
lyrics_rnn_starter/
├── data/
│   ├── raw/          # put your .txt lyric files here (one file per song)
│   └── processed/    # auto-generated corpus + numpy dataset
├── models/           # trained models saved here
├── notebooks/        # optional: your experiments
├── src/
│   ├── preprocess.py
│   ├── train_char_lstm.py
│   └── generate.py
├── requirements.txt
└── README.md
```

## Quick Start

1) **Install deps**
```bash
pip install -r requirements.txt
```

2) **Add data**
- Place `.txt` files in `data/raw/` (avoid copyrighted content in your repo).
- Or edit `data/raw/sample_corpus.txt` (toy text included for testing).

3) **Preprocess**
```bash
python src/preprocess.py --data_dir data/raw --out_dir data/processed --seq_len 120 --step 3
```

4) **Train**
```bash
python src/train_char_lstm.py --data data/processed/train_char.npz --vocab data/processed/vocab.json --epochs 20
```

5) **Generate**
```bash
python src/generate.py --model models/char_lstm.keras --vocab data/processed/vocab.json --seed "tonight i feel" --length 400 --temperature 0.8
```

## Notes
- Start with the **character-level** model to learn the full pipeline.
- Later extensions (not included yet):
  - Word-level model
  - GRU-based model (`--cell gru`)
  - Attention or conditioning by artist/genre
  - Streamlit app for interactive demo

Happy learning! 🎶
