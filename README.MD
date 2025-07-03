# Sentiment Analyzer

This project implements a sentiment analysis system using classical machine learning techniques in Python. It classifies tweets as **positive** or **negative** by applying text preprocessing, feature extraction with TF-IDF, and a Linear Support Vector Classifier.

---

## Overview

The code performs the following operations:

- Loads a tweet dataset from a CSV file
- Cleans and preprocesses each tweet by:
  - removing links, mentions, hashtags, punctuation
  - converting text to lowercase
  - removing stopwords
  - lemmatizing words
- Vectorizes the cleaned text using **TF-IDF** with n-grams up to trigrams
- Splits the dataset into training and testing sets (80%/20%)
- Trains a calibrated LinearSVC classifier for binary sentiment prediction
- Evaluates the model with a classification report and accuracy score
- Provides a function to predict sentiment probabilities for new text inputs

---

## Approach

- **Machine Learning Model:** Linear Support Vector Classifier (LinearSVC) with probability calibration  
- **Feature Extraction:** TF-IDF vectorizer (unigrams, bigrams, trigrams)  
- **Preprocessing Steps:**
  - Lowercasing
  - Removing mentions, hashtags, URLs
  - Removing non-alphabetic characters
  - Removing stopwords
  - Lemmatization
- **Train/Test Split:** 80% training data, 20% testing data  
- **Evaluation:** Classification report and accuracy score

---

## Dataset

The project expects a CSV dataset named `dataSet.csv` with the following columns:

| Column  | Description                                    |
|---------|------------------------------------------------|
| target  | Sentiment label (0 = negative, 4 = positive)   |
| id      |                                                |
| date    |                                                |
| query   |                                                |
| user    |                                                |    
| text    | The tweet text                                 |
Note: Download the Sentiment140 dataset from Kaggle and rename it to dataSet.csv before using it.
---

## How to Run

1. **Clone the project:**

    ```bash
    git clone https://github.com/hafbldprin/SentimentAnalyzer.git
    ```

2. **Navigate to the project folder:**

    ```bash
    cd SentimentAnalyzer
    ```
3. a **Run MakeFile:**
    ```bash
    make
    ```
3. b **Set up a virtual environment:**

    ```bash
    python -m venv venv
    ```

4. **Activate the environment:**

    - **macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

    - **Windows:**

        ```bash
        venv\Scripts\activate
        ```

5. **Install dependencies:**

    ```bash
    pip install pandas scikit-learn nltk
    ```

6. **Place your CSV dataset** in this folder and rename it to `dataSet.csv`.

7. **Train the model:**

    ```bash
    python train.py
    ```

8. **Run the sentiment analyzer:**

    ```bash
    python analyzer.py
    ```

---

## Repository Contents

- `load.py`  
  Loads the CSV dataset, applies filtering and cleaning, and trains the model

- `analyzer.py`  
  Runs the trained model and performs sentiment analysis on new input

- `preprocess.py`
  Helper function for cleaing data set  

- `dataSet.csv`
  Helper function for cleaing data set  

- `requirements.txt`
  List of dependencies (Packages used by this project)  

- `Makeup`  
  Auto-run script to train and run the model

- `README.md`  
  Description of the project

---
