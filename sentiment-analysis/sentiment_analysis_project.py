# -*- coding: utf-8 -*-
"""Sentiment analysis of Amazon Fine Food Reviews.

Compares two approaches on the same reviews:
  1. VADER — a rule-based sentiment scorer from NLTK
  2. RoBERTa — a pretrained transformer model from Hugging Face

Exported from Google Colab; the rendered charts are in
Sentiment_Analysis_Project_Output.pdf next to this file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

plt.style.use('ggplot')

# Load the first 500 reviews (the full dataset has ~500k rows;
# 500 keeps the RoBERTa pass fast enough to run without a big GPU).
# Path is Colab-style — change it if running locally.
df = pd.read_csv('/content/Reviews.csv', engine='python', nrows=500)
print(df.shape)

df.head()

# Reviews per star rating — the dataset is heavily skewed toward 5 stars
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

# --- Basic NLTK pipeline on one example review -----------------------------
# Tokenize, part-of-speech tag, and chunk named entities, to show the
# classic NLP preprocessing steps before any sentiment scoring.

example = df['Text'][50]
print(example)

nltk.download('punkt')
nltk.download('punkt_tab')
tokens = nltk.word_tokenize(example)
tokens[:10]

nltk.download('averaged_perceptron_tagger_eng')
tagged = nltk.pos_tag(tokens)
tagged[:10]

nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

# --- VADER sentiment scoring ------------------------------------------------
# VADER is rule-based: it looks words up in a sentiment lexicon and combines
# the scores. Fast, but blind to context and sarcasm.

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

sia.polarity_scores('I am so happy!')

sia.polarity_scores('This is the worst thing ever.')

sia.polarity_scores(example)

# Score every review. 'compound' is VADER's overall score in [-1, 1].
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

vaders.head()

# If VADER works, its compound score should rise with the star rating
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

# --- RoBERTa (pretrained transformer) ---------------------------------------
# Unlike VADER, the transformer reads the whole sentence in context, so it
# can pick up negation and sarcasm that word-level scoring misses.

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# VADER's take on the example review, for comparison
print(example)
sia.polarity_scores(example)

# RoBERTa's take: raw logits -> softmax -> neg/neu/pos probabilities
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

# Score every review with both models so they can be compared side by side.
# RoBERTa errors on reviews longer than its 512-token limit; skip those.
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# --- Comparing the two models -----------------------------------------------

results_df.columns

# Pairwise scatter of every score column, colored by star rating
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()

# --- Where the models disagree ----------------------------------------------
# 1-star reviews each model scored as most positive, and 5-star reviews
# scored as most negative — the interesting failure cases.

results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]
