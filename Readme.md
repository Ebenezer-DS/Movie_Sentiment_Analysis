**Movie Sentiment Analysis**

This project performs sentiment analysis on a dataset of movie reviews, aiming to classify the sentiment of each review as positive or negative. The goal is to use machine learning techniques to predict the sentiment based on the text data.

**Table of Contents**

__Introduction__
__Data Description__
__Project Structure__
__Setup__
__Modeling__
__Results__
__Conclusion__

**Introduction**

Sentiment analysis is a common task in natural language processing (NLP), where we aim to determine the polarity of textual data. In this project, we analyze a dataset of movie reviews and classify the sentiments as either positive or negative. This classification can help movie producers, critics, and viewers understand audience reactions.

**Data Description**

The dataset used in this project consists of movie reviews, where each review is labeled with its sentiment (positive/negative). The main features include:
__**Review Text**: The raw text of the movie review.__
__**Sentiment**: The sentiment associated with the review (positive or negative).__

**Project Structure**

The project is structured as follows:
├── notebooks/                # Jupyter notebooks with analysis
├── models/                   # Trained machine learning models
├── README.md                 # Project description
├── requirements.txt          # Dependencies required to run the project
├── app.py  # My streamlit application
├── sentiment_model.h5  # Your trained model
└── tokenizer.pickle  # Your tokenizer

**Setup**

**Dependencies**

To install the required dependencies, run:
pip install -r requirements.txt

The main libraries used include:

__Python 3.12__
__scikit-learn__
__pandas__
__numpy__
__matplotlib__
__nltk__

**Running the Project**

1. Clone the repository:
git clone https://github.com/Ebenezer-DS/movie-sentiment-analysis.git
cd movie-sentiment-analysis

2. Ensure all dependencies are installed: Make sure you have installed the necessary libraries by running:
pip install -r requirements.txt

3. Run the Streamlit app: The entire workflow (preprocessing, model training, evaluation, and prediction) is now handled by app.py. To launch the app locally, simply run:
streamlit run app.py

4. Push changes to GitHub: Once you’ve made changes or added files, you can push them to GitHub:
git add .
git commit -m "Updated Streamlit app"
git push origin main

**Modeling**
The project uses various machine learning algorithms, including:
__Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture specifically designed to handle long-term dependencies in sequential data. Unlike traditional RNNs, which struggle with the vanishing gradient problem, LSTMs use special "memory cells" and gates (input, forget, and output gates) that control the flow of information, allowing them to retain information over longer sequences. This makes LSTMs particularly effective for tasks like natural language processing, time-series forecasting, and any other task involving sequential data. In sentiment analysis, LSTMs can capture the context and sentiment expressed in a sequence of words, making them highly effective for text classification.__

The models are trained on a portion of the data and tested on a separate validation set to evaluate performance.

**Results**
The models are evaluated using common metrics such as accuracy, precision, recall, and F1-score. The results indicate that the sentiment classification can be performed with reasonable accuracy using the chosen models.

Model	                   Accuracy	Precision	Recall	F1-score
LSTM                        0.8790    0.8585     0.9116   0.8843

**Conclusion**

This project demonstrates how sentiment analysis can be applied to movie reviews to classify sentiment using machine learning models. By preprocessing the text data and training models, we can achieve accurate predictions of movie review sentiment. Long Short-Term Memory (LSTM) models, a type of deep learning model, are especially useful for this task as they are well-suited for processing sequential data like text.