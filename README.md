# NLP Model for Text Analytics and Classification

This repository contains the implementation of an NLP-based text classification model designed to categorize complaints based on various parameters such as the victim, type of fraud, and other relevant details. The model preprocesses textual data, extracts important features, and classifies complaints into predefined categories.

## Objective

The goal of this project is to build a Natural Language Processing (NLP) model that can:
- **Preprocess and clean text data** from complaint records.
- **Classify complaints** based on the victim, type of fraud, and other relevant attributes.
- **Evaluate the performance** of the model using standard classification metrics.

## Features

### 1. **Data Manipulation and Visualization**
- **pandas**: Provides data structures like DataFrames, which are essential for handling and manipulating the dataset.
- **numpy**: Used for numerical computations and working with arrays.
- **matplotlib** & **seaborn**: For creating visualizations such as bar plots, histograms, and heatmaps to analyze the data.

### 2. **Text Preprocessing**
- **nltk** (Natural Language Toolkit): A library used for a variety of text preprocessing tasks such as:
  - **Tokenization**: Splitting text into words or tokens.
  - **Stopword Removal**: Filtering out common words like "the", "is", etc., that do not carry much meaning.
  - **Stemming**: Reducing words to their base form (e.g., "running" → "run") using **SnowballStemmer**.
  - **Lemmatization**: Converting words to their base form using **WordNetLemmatizer** (e.g., "better" → "good").
  - **POS (Part-of-Speech) Tagging**: Assigning grammatical tags to words (e.g., noun, verb).

  Downloaded NLTK resources include:
  - **punkt**: Pre-trained tokenizer for splitting text into sentences and words.
  - **averaged_perceptron_tagger**: A POS tagger.
  - **wordnet**: A lexical database for English, used in lemmatization.

- **re** and **string**: Python's built-in libraries for text cleaning tasks, such as removing unwanted characters or punctuation from the text.

### 3. **Model Building**
- **scikit-learn**: A comprehensive library for machine learning algorithms and tools, used for model building and evaluation:
  - **train_test_split**: Splitting the data into training and test sets.
  - **LogisticRegression**, **SGDClassifier**, **MultinomialNB**: Various machine learning models for text classification.
  - **classification_report**, **f1_score**, **accuracy_score**, **confusion_matrix**: Evaluation metrics to assess the model’s performance.
  - **roc_curve**, **roc_auc_score**: For plotting ROC curves and calculating the AUC score (useful for binary classification tasks).
  
- **TfidfVectorizer** & **CountVectorizer**: Used to convert text into numerical features for classification using the Bag-of-Words (BoW) model or TF-IDF approach. Both are commonly used to extract important features from the text.

### 4. **Word Embedding**
- **gensim**: A library focused on unsupervised learning, specifically for generating word embeddings and modeling word relationships. **Word2Vec** is used to generate vector representations for words in a high-dimensional space, which can capture semantic relationships and meanings. This can be useful for deep learning-based models or for handling large datasets.

#### Key Functions and Features:
- **Tokenization**: Converting sentences into individual words or tokens.
- **Stopword Removal**: Removing words that don’t provide meaningful information.
- **Stemming and Lemmatization**: Reducing words to their root form for better matching.
- **Feature Extraction**: Converting text into a numerical representation (e.g., using TF-IDF or Count Vectorizer).
- **Model Training**: Applying machine learning algorithms (e.g., Logistic Regression, Naive Bayes, etc.) for text classification.
- **Model Evaluation**: Using metrics like F1-score, accuracy, and confusion matrix to evaluate the model's performance.

### 5. **Word2Vec Embeddings**
- The **Word2Vec** model from **gensim** is used for learning word embeddings on large corpora. **Word2Vec** captures semantic similarity between words (e.g., "king" and "queen" are closer in the vector space compared to "king" and "apple"). This can be used to enhance the classification model, especially when working with larger datasets.
