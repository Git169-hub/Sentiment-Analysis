# Text Sentiment & Spam Classifier — DistilBERT

Two fine-tuned DistilBERT models for text classification deployed
as an interactive Gradio app on HuggingFace Spaces.

## Live Demo

👉 [Try it here](https://huggingface.co/spaces/RazakAIhub/sentiment-spam-classifier)

## Models

| Model | Task | Accuracy |
|---|---|---|
| [distilbert-fake-news-classifier](https://huggingface.co/RazakAIhub/distilbert-fake-news-classifier) | Fake vs Real news | 99.15% |
| [distilbert-spam-classifier](https://huggingface.co/RazakAIhub/distilbert-spam-classifier) | Spam detection | — |

## Architecture

Input text → DistilBERT tokenizer (max 512 tokens)
→ DistilBERT encoder → CLS token → classification head
→ FAKE/REAL or SPAM/NOT SPAM + confidence score

## Tech Stack

| Component | Tool |
|---|---|
| Models | DistilBERT (distilbert-base-uncased) |
| Framework | PyTorch + HuggingFace Transformers |
| Training | Google Colab T4 GPU |
| Deployment | Gradio on HuggingFace Spaces |

## How to Use

```python
from transformers import pipeline

# Fake news detection
pipe = pipeline(
    "text-classification",
    model="RazakAIhub/distilbert-fake-news-classifier"
)
pipe("NASA confirms water found on Mars.")
# [{'label': 'REAL', 'score': 0.997}]

# Spam detection
pipe = pipeline(
    "text-classification", 
    model="RazakAIhub/distilbert-spam-classifier"
)
pipe("You have won a $1000 gift card! Click now!")
# [{'label': 'SPAM', 'score': 0.991}]
```

## How to Run Locally

```bash
git clone https://github.com/Git169-hub/Sentiment-Analysis
cd Sentiment-Analysis
pip install transformers torch gradio
python app.py
```

## Author

Razak Shaik | VIT-AP University | CS Final Year

| [Sentiment & Spam Classifier](https://github.com/Git169-hub/Sentiment-Analysis) | Fake news + spam detection | DistilBERT, HuggingFace, Gradio | [Try it](https://huggingface.co/spaces/RazakAIhub/sentiment-spam-classifier) |

[HuggingFace](https://huggingface.co/RazakAIhub) | 
[LinkedIn](https://www.linkedin.com/in/shaik-razak-6493b7257) | 
[GitHub](https://github.com/Git169-hub)
