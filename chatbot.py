import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

import pandas as pd
import re
import joblib
import numpy as np
from collections import defaultdict
from math import log

PAGE_COMMANDS = ['next', 'prev', 'first', 'last']
EXIT_COMMANDS = ['exit', 'thank you', 'bye']

def preprocess_text(text):
    return re.sub(r'[^\w\s]', '', text.lower())

class CustomNaiveBayes:
    def __init__(self):
        self.class_prior = {}
        self.feature_prob = {}

    def fit(self, X, y):
        num_docs, num_features = X.shape
        unique_classes, counts = np.unique(y, return_counts=True)
        self.class_prior = {cls: count / num_docs for cls, count in zip(unique_classes, counts)}

        feature_sum_by_class = {cls: np.zeros(num_features) for cls in unique_classes}
        for cls in unique_classes:
            X_cls = X[y == cls]
            feature_sum_by_class[cls] = X_cls.sum(axis=0) + 1

        total_features_per_class = {cls: np.sum(feature_sum_by_class[cls]) for cls in unique_classes}
        self.feature_prob = {
            cls: feature_sum_by_class[cls] / total_features_per_class[cls]
            for cls in unique_classes
        }

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for cls in self.class_prior:
                log_prior = log(self.class_prior[cls])
                log_likelihood = sum(x * np.log(self.feature_prob[cls]))
                class_scores[cls] = log_prior + log_likelihood
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)
    
class CustomTFIDF:
    def __init__(self):
        self.idf = {}
        self.word_to_index = {}
        self.index_to_word = {}

    def fit_transform(self, corpus):
        num_docs = len(corpus)
        term_doc_count = defaultdict(int)
        tokenized_corpus = []

        for doc in corpus:
            tokens = doc.split()
            tokenized_corpus.append(tokens)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_count[token] += 1

        for term, count in term_doc_count.items():
            self.idf[term] = log((num_docs + 1) / (count + 1)) + 1

        self.word_to_index = {word: idx for idx, word in enumerate(self.idf)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

        tfidf_matrix = []
        for tokens in tokenized_corpus:
            doc_vector = np.zeros(len(self.word_to_index))
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1
            for term, count in term_count.items():
                if term in self.word_to_index:
                    idx = self.word_to_index[term]
                    tf = count / len(tokens)
                    doc_vector[idx] = tf * self.idf[term]
            tfidf_matrix.append(doc_vector)
        return np.array(tfidf_matrix)

    def transform(self, corpus):
        tfidf_matrix = []
        for doc in corpus:
            tokens = doc.split()
            doc_vector = np.zeros(len(self.word_to_index))
            term_count = defaultdict(int)
            for token in tokens:
                term_count[token] += 1
            for term, count in term_count.items():
                if term in self.word_to_index:
                    idx = self.word_to_index[term]
                    tf = count / len(tokens)
                    doc_vector[idx] = tf * self.idf.get(term, 0)
            tfidf_matrix.append(doc_vector)
        return np.array(tfidf_matrix)

class CustomLabelEncoder:
    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}

    def fit_transform(self, labels):
        unique_labels = set(labels)
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        return np.array([self.label_to_index[label] for label in labels])

    def inverse_transform(self, indices):
        return np.array([self.index_to_label[idx] for idx in indices])

def load_models():
    try:
        # model = joblib.load(r'saved_models\Naive Bayes_model.pkl')
        # vectorizer = joblib.load(r'saved_models\vectorizer.pkl')
        # label_encoder = joblib.load(r'saved_models\label_encoder.pkl')
        model = joblib.load(r'saved_custom_models\Naive Bayes_custom_model.pkl')
        vectorizer = joblib.load(r'saved_custom_models\custom_vectorizer.pkl')
        label_encoder = joblib.load(r'saved_custom_models\custom_label_encoder.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}")
        exit(1)
    return {"model": model, "vectorizer": vectorizer, "label_encoder": label_encoder}

def predict_genres(input_text, models):
    input_text_processed = preprocess_text(input_text)
    input_vec = models['vectorizer'].transform([input_text_processed])
    predicted_encoded = models['model'].predict(input_vec)
    return models['label_encoder'].inverse_transform(predicted_encoded)[0]

def extract_genres_from_input(user_input, models, valid_genres):
    user_input_processed = preprocess_text(user_input)
    words = user_input_processed.split()

    predicted_genres = set()
    for word in words:
        if word in ['suggest', 'films', 'and', 'but']: continue
        predicted_genre = predict_genres(word, models)
        if predicted_genre in valid_genres:
            predicted_genres.add(predicted_genre)
    return list(predicted_genres)

def normalize_popularity_within_genre(filtered_movies):
    min_popularity, max_popularity = filtered_movies['popularity'].min(), filtered_movies['popularity'].max()
    filtered_movies['normalized_popularity'] = (filtered_movies['popularity'] - min_popularity) / (max_popularity - min_popularity)
    return filtered_movies

def load_movie_data():
    try:
        df_movies = pd.read_csv('movie_dataset.csv')
        df_movies['genres'] = df_movies['genres'].fillna('').str.lower()
        df_movies['release_date'] = pd.to_datetime(df_movies['release_date'], format='%Y-%m-%d', errors='coerce')
        df_movies['release_year'] = df_movies['release_date'].dt.year.astype('Int64')
        df_movies['revenue_to_budget_ratio'] = df_movies['revenue'] / df_movies['budget']
        df_movies['runtime'] = df_movies['runtime'].fillna(0)
    except FileNotFoundError as e:
        print(f"Error loading movie data: {e}")
        exit(1)
    return df_movies

def recommend_movies_by_genre(genres, top_n=5, page=1, df_movies=None):
    if not genres:
        return "No valid genres found in your input.", None

    genre_filtered_movies = df_movies[df_movies['genres'].apply(lambda x: all(genre in x.split(' ') for genre in genres))].copy()
    if genre_filtered_movies.empty:
        return f"No movies found with the genre(s): {', '.join(genres)}.", None

    global_average = genre_filtered_movies['vote_average'].mean()
    m = genre_filtered_movies['vote_count'].mean()

    genre_filtered_movies['smoothed_score'] = (
        genre_filtered_movies['vote_count'] / (genre_filtered_movies['vote_count'] + m) * genre_filtered_movies['vote_average'] +
        m / (genre_filtered_movies['vote_count'] + m) * global_average
    )

    genre_filtered_movies = normalize_popularity_within_genre(genre_filtered_movies)
    genre_filtered_movies['final_score'] = 0.7 * genre_filtered_movies['smoothed_score'] + 0.3 * genre_filtered_movies['normalized_popularity']
    genre_filtered_movies['genre_count'] = genre_filtered_movies['genres'].apply(lambda x: len(x.split(' ')))

    genre_filtered_movies = genre_filtered_movies.sort_values(
        by=['final_score', 'genre_count', 'revenue_to_budget_ratio', 'release_year', 'runtime'],
        ascending=[False, True, False, False, True]
    )

    total_pages = (genre_filtered_movies.shape[0] - 1) // top_n + 1
    start_idx = (page - 1) * top_n
    end_idx = start_idx + top_n
    page_movies = genre_filtered_movies.iloc[start_idx:end_idx]

    if page_movies.empty:
        return f"No more movies to display on page {page}.", None

    return page_movies[['title', 'genres', 'release_year', 'vote_average', 'vote_count']], total_pages

def handle_navigation_commands(user_input, current_page, total_pages):
    if user_input.lower() == 'next':
        current_page = min(current_page + 1, total_pages)
    elif user_input.lower() == 'prev':
        current_page = max(current_page - 1, 1)
    elif user_input.lower() == 'first':
        current_page = 1
    elif user_input.lower() == 'last':
        current_page = total_pages
    return current_page

def start_chat():
    current_page, genres, total_pages = 1, [], 1
    models = load_models()
    df_movies = load_movie_data()
    valid_genres = set(df_movies['genres'].str.split(' ').explode().unique())

    print("Welcome to the Movie Recommendation Chatbot!")

    while True:
        user_input = input("You: ")

        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break

        if user_input.lower() in PAGE_COMMANDS:
            current_page = handle_navigation_commands(user_input, current_page, total_pages)
            movie_recommendations, total_pages = recommend_movies_by_genre(genres, page=current_page, df_movies=df_movies)
            print(f"Page {current_page} of {total_pages}:\n", movie_recommendations)

        else:
            genres = extract_genres_from_input(user_input, models, valid_genres)
            if genres:
                current_page = 1
                movie_recommendations, total_pages = recommend_movies_by_genre(genres, page=current_page, df_movies=df_movies)
                print(f"Page 1 of {total_pages}:\n", movie_recommendations)
            else:
                print("No genres detected. Please try again.")

start_chat()
