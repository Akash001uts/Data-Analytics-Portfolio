# Data Analytics Portfolio — Akash Bhatnagar

Three end-to-end data projects covering unsupervised learning, NLP, and time-series forecasting.

| Project | Folder | Techniques |
|---|---|---|
| Customer Segmentation | [`customer-segmentation/`](customer-segmentation) | K-Means clustering, elbow method |
| Sentiment Analysis | [`sentiment-analysis/`](sentiment-analysis) | VADER, RoBERTa (Hugging Face) |
| Stock Forecasting | [`stock-forecasting/`](stock-forecasting) | LSTM networks (TensorFlow/Keras) |

The datasets used by all three projects are bundled in [`data/Datasets.zip`](data/Datasets.zip) — unzip it and place the CSVs next to the notebook you're running (the notebooks were written in Google Colab, so their paths point at `/content/`; adjust the path if you run them locally).

## 1) Customer Segmentation (Unsupervised Learning)

* **Goal:** Segment mall customers into distinct groups based on Annual Income vs. Spending Score, to inform marketing strategy.
* **Dataset:** Mall Customers (`Mall_Customers.csv`) — 200 records of customer demographics and spending behavior.
* **Method:** Elbow method (WCSS for k = 1–10) to pick the cluster count, then K-Means with `n_clusters=5`.
* **Key results:** Five clear customer personas (e.g. High Income / Low Spending, Low Income / High Spending), visualized with the cluster centroids on a 2D scatter plot.
* **Run:** open `customer-segmentation/Customer_Segmentation_Project.ipynb` with `Mall_Customers.csv` alongside it.

## 2) Sentiment Analysis (NLP)

* **Goal:** Classify customer sentiment in product reviews, comparing a rule-based approach against a transformer model.
* **Dataset:** Amazon Fine Food Reviews (`Reviews.csv`), first 500 reviews.
* **Method:** VADER for rule-based polarity scoring, then `cardiffnlp/twitter-roberta-base-sentiment` (Hugging Face) for context-aware scoring; results compared per star rating.
* **Key results:** RoBERTa clearly outperforms VADER on sarcastic and complex negative reviews. See the rendered charts in `sentiment-analysis/Sentiment_Analysis_Project_Output.pdf`.
* **Run:** `sentiment-analysis/sentiment_analysis_project.py` (Colab export — needs `nltk` and `transformers`; a GPU helps for RoBERTa).

## 3) Stock Forecasting with LSTMs (Time Series)

* **Goal:** Predict Microsoft (MSFT) closing prices from historical data (1986–2022).
* **Method:** Two LSTM models — a 3-day sliding window model with train/validation/test splits and a recursive-forecasting experiment, then a deeper 60-day-window model with dropout.
* **Key results:** The models track held-out test data closely; the recursive experiment shows how prediction error compounds without fresh observations.
* **Run:** open `stock-forecasting/Stock_Forecasting_with_LSTMs.ipynb` with `MSFT.csv` / `MicrosoftStock.csv` alongside it.

## Tech Stack

Python 3 · pandas · numpy · matplotlib · seaborn · scikit-learn · TensorFlow (Keras) · NLTK · Hugging Face Transformers
