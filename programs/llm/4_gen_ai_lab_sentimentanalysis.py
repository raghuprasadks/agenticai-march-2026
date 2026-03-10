#pip install  torch transformers
from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]

# Example sentences for sentiment analysis
sentences = [
    "I love the new AI advancements, they are truly revolutionary!",
    "The new software update is frustrating and full of bugs.",
    "Customer service was amazing, very helpful and responsive.",
    "I'm disappointed with the product quality, not what I expected.",
    "This experience has been wonderful, highly recommend!"
]

# Analyze sentiment for each sentence
for sentence in sentences:
    sentiment_result = analyze_sentiment(sentence)
    print(f"Sentence: {sentence}\nSentiment: {sentiment_result}\n")

