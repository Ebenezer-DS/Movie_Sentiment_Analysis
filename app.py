import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import zipfile

# Load the model and tokenizer
model = tf.keras.models.load_model('sentiment_model.h5', compile=False)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100  # Max sequence length

# Function to clean review text
def clean_review_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Function to generate visualizations
def generate_visualizations(df):
    with st.expander("Sentiment Distribution"):
        st.subheader('Sentiment Distribution')
        df['sentiment'] = df['sentiment'].str.lower()  # Normalize case for sentiments
        df['sentiment'].value_counts().plot(kind='bar', color=['#4CAF50', '#FF5722'])  # Green for positive, red-orange for negative
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(plt.gcf())
        plt.clf()

    with st.expander("Review Length Distribution"):
        st.subheader('Review Length Distribution')
        df['review_length'] = df['review'].apply(lambda x: len(x.split()))
        plt.hist(df['review_length'], bins=20, color='#FFC107', edgecolor='black')  # Yellow with black edges for clarity
        plt.title('Review Length Distribution')
        plt.xlabel('Length of Review (Words)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt.gcf())
        plt.clf()

    with st.expander("Cleaned Review Length Distribution"):
        st.subheader('Cleaned Review Length Distribution')
        df['cleaned_review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
        plt.hist(df['cleaned_review_length'], bins=20, color='#2196F3', edgecolor='black')  # Blue with black edges
        plt.title('Cleaned Review Length Distribution')
        plt.xlabel('Length of Cleaned Review (Words)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt.gcf())
        plt.clf()

# Generate word cloud
def generate_word_distribution(clean_text):
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap='inferno').generate(clean_text)  # Inferno colormap for striking visuals
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

# Streamlit app
st.title('Movie Sentiment Prediction')

# Input form for user review
input_data = st.text_area("Enter your movie review", height=150)

# Add button above visualizations
if st.button('Predict Sentiment'):
    clean_text = clean_review_text(input_data)
    sequences = tokenizer.texts_to_sequences([clean_text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Predict sentiment
    prediction = model.predict(padded_sequence)
    
    # Display prediction probabilities for better transparency
    st.write(f"Prediction Probability: {prediction[0][0]:.4f}")

    # Convert the prediction to a class label based on a threshold
    threshold = 0.58
    predicted_class = (prediction > threshold).astype("int32")

    # Convert the predicted class to sentiment
    sentiment = "positive" if predicted_class[0][0] == 1 else "negative"

    # Display result with consistent case
    st.write(f"Sentiment: **{sentiment.capitalize()}**")

    # Generate word cloud for cleaned text
    generate_word_distribution(clean_text)

# Load the CSV file from a ZIP archive and generate visualizations
with zipfile.ZipFile('reviews_data.zip', 'r') as z:
    with z.open('reviews_data.csv') as f:
        df = pd.read_csv(f)

generate_visualizations(df)
