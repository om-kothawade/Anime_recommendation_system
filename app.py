from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the preprocessed data and similarity matrix
with open('preprocessed_data.pkl', 'rb') as f:
    new_df = pickle.load(f)
with open('similarity_matrix.pkl', 'rb') as f:
    similarity = pickle.load(f)

def recommend(anime):
    if anime not in new_df['Name'].values:
        return None

    anime_index = new_df[new_df['Name'] == anime].index[0]
    distances = similarity[anime_index]
    anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendations = []

    for i in anime_list:
        row = new_df.iloc[i[0]]
        recommendations.append({
            'Name': row['Name'],
            'Genres': ', '.join(row['Genres']),
            'Episodes': row['Episodes'],
            'Score': row['Score'],
            'Type': row['Type'],
            'Rating': row['Rating']
        })
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    anime_name = request.form['anime_name']
    recommendations = recommend(anime_name)
    return render_template('index.html', recommendations=recommendations, anime_name=anime_name)

if __name__ == '__main__':
    app.run(debug=True)
