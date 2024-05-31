import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import pickle

# Load and preprocess the dataset
anime_df = pd.read_csv('/Users/omkothawade/Documents/Anime recommedation website/anime-dataset-2023.csv')
anime_df = anime_df[['anime_id', 'Name', 'Synopsis', 'Genres', 'Episodes', 'Score', 'Type', 'Rating']]
anime_df['Synopsis'] = anime_df['Synopsis'].apply(lambda x: x.split())
anime_df['Genres'] = anime_df['Genres'].apply(lambda x: x.split(','))
anime_df['Genres'] = anime_df['Genres'].apply(lambda x: [i.replace(" ", "") for i in x])
anime_df['tags'] = anime_df['Synopsis'] + anime_df['Genres']
new_df = anime_df[['anime_id', 'Name', 'Episodes', 'Score', 'Type', 'Rating', 'tags', 'Genres']]
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Stemming
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)

# Vectorization and similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# Save the preprocessed data and similarity matrix
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(new_df, f)
with open('similarity_matrix.pkl', 'wb') as f:
    pickle.dump(similarity, f)
