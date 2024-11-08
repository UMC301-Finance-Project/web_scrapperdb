from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline
from torch import nn
import torch
import torch.nn.functional as F
# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification \
  .from_pretrained("yangheng/deberta-v3-base-absa-v1.1")

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load a traditional Sentiment Analysis model
sentiment_model_path = "ProsusAI/finbert"
sentiment_model = pipeline("sentiment-analysis", model=sentiment_model_path,
                          tokenizer=sentiment_model_path, device=device)
aspects = [
    "Earnings",
    "Revenue",
    "Margins",
    "Dividend",
    "EBITDA",
    "Debt",
    "Sentiment"
]


class SentimentAnalyser:
    def __init__(self, chaspects=aspects):
        self.absa_model = absa_model
        self.sentiment_model = sentiment_model
        self.aspects = chaspects
        self.absa_tokenizer = absa_tokenizer

    def analyze_sentiment(self, sentence):
        """Analyze sentiment for each aspect and return a signed score based on positive, neutral, or negative dominance."""
        aspect_scores = {}

        for aspect in self.aspects:
            # Prepare input for the ABSA model
            inputs = self.absa_tokenizer(f"[CLS] {sentence} [SEP] {aspect} [SEP]", return_tensors="pt")
            outputs = self.absa_model(**inputs)
            
            # Get probabilities for negative, neutral, and positive sentiments
            probs = F.softmax(outputs.logits, dim=1).detach().numpy()[0]
            negative_prob, neutral_prob, positive_prob = probs

            # Determine the score based on the highest probability
            if neutral_prob > max(positive_prob, negative_prob):
                score = 0  # Neutral sentiment dominates
            elif positive_prob > negative_prob:
                score = float(positive_prob)  # Positive sentiment dominates
            else:
                score = float(-1*negative_prob)  # Negative sentiment dominates

            # Store the score for the current aspect
            aspect_scores[aspect] = score
            if aspect=="Sentiment":
                asf = self.sentiment_model([sentence])[0]
                if asf['label']=='neutral':
                    prob=0
                elif asf['label'] =='positive':
                    prob= float(asf['score'])
                else:
                    prob = float(-1*asf['score'])

                aspect_scores['Sentiment'] =prob
        return aspect_scores