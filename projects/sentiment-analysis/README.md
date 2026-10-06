# Sentiment analysis, redone

**Status:** planned. I'll start this once the hospital admissions project is written up.

My first sentiment project compared VADER (a rule-based scorer) with a RoBERTa transformer on Amazon Fine Food Reviews,
and concluded that RoBERTa was clearly better. Looking back, I never actually measured that. I plotted each model's
scores against the star ratings and looked at a few example reviews, but I never calculated an accuracy or F1
score. I also only used 500 reviews, dropped any review that was too long instead of truncating it, and used a model
trained on tweets for food reviews. So this time I want to answer the question properly.

## The question

When you measure it properly, how much better is a transformer than simple baselines at reading the sentiment of food
reviews, and where do they all go wrong?

## Plan

- **Data:** the full Amazon Fine Food Reviews dataset (about 568,000 reviews), downloaded by script rather than
  committed to the repo.
- **Labels:** 1 to 2 stars negative, 3 neutral, 4 to 5 positive.
- **Deduplicate before splitting.** The same review text often appears under several products. If one copy lands in
  training and another in testing, the test score is inflated. I'll dedupe on the reviewer and the review text first,
  and add a test that no text appears in more than one split.
- **Models:**
  - VADER, as before;
  - TF-IDF with logistic regression, a simple baseline trained on its own split;
  - the RoBERTa sentiment model, this time with long reviews truncated instead of dropped.
- **Evaluation:** macro-F1 (so the smaller neutral class counts as much as the others), precision and recall per class,
  confusion matrices, and bootstrap confidence intervals.
- **Error analysis:** how each model does on long reviews, mixed reviews ("great taste, awful packaging"), and
  sarcasm.
- Transformer outputs get cached, so the tests and notebooks don't need a GPU.
