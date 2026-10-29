from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

# Load pretrained FinBERT
model_name = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sentiment labels, in the model's own output order (finbert: positive, negative, neutral)
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

print("\n=== FinBERT Sentiment Analyzer ===")

while True:

    text = input("\nEnter financial headline: ")

    if text.lower() == "exit":
        break

    # Convert text into tensors
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to probabilities
    predictions = torch.nn.functional.softmax(
        outputs.logits,
        dim=-1
    )

    # Get highest probability class
    predicted_class = torch.argmax(predictions).item()


    confidence = predictions[0][predicted_class].item()

    # Print result
    print("\nSentiment:", labels[predicted_class])
    print("Confidence:", round(confidence, 4))
