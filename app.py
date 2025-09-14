import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set wide layout
st.set_page_config(layout="wide")

# Load model components
vectorizer = joblib.load("vectorizer.pkl")
svd = joblib.load("svd.pkl")
voting = joblib.load("prediction.pkl")

# Title
st.title("🌟 Product Review Sentiment Analysis Dashboard")

# --- Section 1: User Input ---
st.header("✍️ Input Your Review")
user_review = st.text_area("Enter your product review here:")

if st.button("Predict Sentiment"):
    if user_review.strip() != "":
        X_vec = vectorizer.transform([user_review])
        X_svd = svd.transform(X_vec)
        pred = voting.predict(X_svd)

        sentiment_map = {
            1: 'Very Negative',
            2: 'Negative',
            3: 'Neutral',
            4: 'Positive',
            5: 'Very Positive'
        }

        sentiment = sentiment_map[pred[0]]
        st.success(f"✅ Predicted Sentiment: **{sentiment}**")
    else:
        st.warning("⚠️ Please enter a review.")

# --- Section 2: Batch Predictions Example ---
st.header("🔁 Sample Reviews Batch Prediction")

new_reviews = [
    "The product was amazing and worked perfectly!",
    "Terrible experience, the product broke after two days!",
    "It's ok but not as good as I expected.",
    "Not expected result from this product",
    "Somewhat okay product"
]

X_vec_batch = vectorizer.transform(new_reviews)
X_svd_batch = svd.transform(X_vec_batch)
batch_preds = voting.predict(X_svd_batch)

sentiment_map = {1: 'Very Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Very Positive'}
batch_sentiments = [sentiment_map[p] for p in batch_preds]

batch_df = pd.DataFrame({
    "Review": new_reviews,
    "Predicted Sentiment": batch_sentiments
})
st.dataframe(batch_df)

# --- Section 3: Sentiment Distribution ---
st.header("📊 Sentiment Distribution (Sample Data)")
sample_data = pd.DataFrame({
    "Sentiment": np.random.choice(batch_sentiments, 100)
})

sentiment_counts = sample_data["Sentiment"].value_counts()

plt.figure(figsize=(10, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment")
plt.ylabel("Count")
st.pyplot(plt)

# --- Section 4: Word Cloud ---
st.header("☁️ Word Cloud from Sample Sentiments")
wordcloud_data = " ".join(sample_data["Sentiment"])
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="plasma").generate(wordcloud_data)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
st.pyplot(plt)
