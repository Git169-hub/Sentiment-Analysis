# Text Sentiment Classifier — IMDB Reviews

## Problem
Classify movie reviews as positive or negative sentiment.

## Dataset
IMDB — 25,000 train, 25,000 test reviews (HuggingFace datasets)

## Architecture
- Embedding layer (vocab 10k, dim 128)
- Bidirectional GRU (2 layers, hidden 128)
- Dropout 0.3 + gradient clipping
- BCEWithLogitsLoss

## Results
- **Val Accuracy: 86%**
- Precision: 0.86 | Recall: 0.86 | F1: 0.86
- Balanced performance on both classes

## Key Learnings
- Word embeddings convert tokens to dense vectors the GRU can process
- Bidirectional GRU reads sequences forward and backward simultaneously
- Model showed overfitting after epoch 4 — best checkpoint saved at epoch 4
- Neutral/ambiguous reviews get low confidence scores — correct behavior

## Tech Stack
Python | PyTorch | HuggingFace Datasets | NumPy | Matplotlib | scikit-learn
