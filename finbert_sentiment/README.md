# FinBERT headline sentiment

Interactive sentiment classifier for financial headlines using the pretrained
[ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) model.

```bash
cd finbert_sentiment
uv run python main.py
```

Type a headline to get `positive` / `negative` / `neutral` and the model's confidence; type `exit` to quit.
The first run downloads the model (~440 MB).

Labels are read from the model's own config (its output order is positive, negative, neutral).

FinBERT handles general finance language well but misreads Indian banking terms, e.g.
"PNB NPAs jump to 12%" scores positive. The sentiment agent (`../sentiment_agent`) can use it as an
optional backend (`SENTIMENT_BACKEND=finbert`); its default is the TF-IDF model.
