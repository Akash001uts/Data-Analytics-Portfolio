# Data Analytics Portfolio (Akash Bhatnagar)

## Featured Projects

### 1) Customer Segmentation (Unsupervised Learning)
* **Goal:** To segment customers into distinct groups based on purchasing habits (Annual Income vs. Spending Score) to optimize marketing strategies.
* **Dataset:** Mall Customers Dataset (`Mall_Customers.csv`) – Contains 200 records of customer demographics and spending behavior.
* **Method:**
    * **Elbow Method:** Calculated Within-Cluster Sum of Square (WCSS) for k=1 to 10 to determine the optimal number of clusters.
    * **K-Means Clustering:** Applied the K-Means algorithm with `n_clusters=5` to categorize customers.
* **Key results:**
    * Identified **5 distinct customer personas** (e.g., High Income/Low Spending, Low Income/High Spending).
    * Visualized the clusters and their centroids using a 2D scatter plot with Seaborn/Matplotlib.
* **How to run:**
  1. Ensure `Mall_Customers.csv` is in the root directory.
  2. Run `Customer_Segmentation_Project.ipynb`.

### 2) Sentiment Analysis
* **Goal:** To analyze and classify customer sentiment from text reviews, comparing traditional "Bag of Words" methods against modern Deep Learning techniques.
* **Dataset:** Amazon Fine Food Reviews (`Reviews.csv`) – A dataset of ~500,000 reviews.
* **Method:**
    * **VADER (Valence Aware Dictionary and sEntiment Reasoner):** Used for rule-based sentiment scoring.
    * **RoBERTa (Hugging Face Transformers):** Implemented a pre-trained Transfer Learning model to capture context and nuance that VADER missed.
* **Key results:**
    * The RoBERTa model significantly outperformed VADER in detecting sarcasm and complex negative reviews.
    * Visualized the distribution of sentiment scores across 1-star to 5-star ratings using Seaborn.
* **How to run:**
  1. Install dependencies (specifically `transformers` and `nltk`).
  2. Run `Sentiment_Analysis_Project.ipynb`.
  3. *Note:* The RoBERTa model requires significant compute power; a GPU is recommended.

### 3) Stock Forecasting with LSTMs
* **Goal:** To predict future stock prices of Microsoft (MSFT) using historical time-series data.
* **Dataset:** MSFT historical stock data spanning 1986 to 2022.
* **Method:**
    * **Preprocessing:** Handled datetime indexing and performed MinMax scaling on "Close" prices.
    * **Model:** Built and trained a Long Short-Term Memory (LSTM) Recurrent Neural Network using TensorFlow/Keras, optimized for sequence prediction.
* **Key results:**
    * The model successfully learned long-term dependencies in price trends over a 30-year period.
    * Generated visualizations comparing predicted prices against actual historical data to validate accuracy.
* **How to run:**
  1. Run `Stock_Forecasting_with_LSTMs.ipynb`.
  2. The notebook will automatically download/load the dataset and train the LSTM model.

## Tech Stack
* **Languages:** Python 3.x
* **Data Manipulation:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Machine Learning & Deep Learning:** TensorFlow (Keras), scikit-learn
* **NLP:** NLTK, Transformers (Hugging Face)
