import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# Get backend file location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_DIR = os.path.join(BASE_DIR, "ML", "Models")

# Load saved recommendation artifacts
tfidf = joblib.load(os.path.join(MODEL_DIR, "recommendation_vectorizer.pkl"))
tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, "recommendation_job_vectors.pkl"))
data = joblib.load(os.path.join(MODEL_DIR, "recommendation_jobs_data.pkl"))

data = data.fillna("")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def recommend_jobs(user_input, top_n=5, location_filter="", experience_filter=""):

    user_input = clean_text(user_input)

    user_vector = tfidf.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Get top 50 most similar jobs first
    top_indices = similarity_scores.argsort()[-50:][::-1]

    filtered_results = []

    for idx in top_indices:
        job = data.iloc[idx]

        if location_filter and location_filter.lower() not in str(job['location']).lower():
            continue

        if experience_filter and experience_filter.lower() not in str(job['required_experience']).lower():
            continue

        filtered_results.append((idx, similarity_scores[idx]))

        if len(filtered_results) == top_n:
            break

    if not filtered_results:
        return []

    recommended = data.iloc[[i[0] for i in filtered_results]][
        ['title', 'location', 'industry', 'required_experience']
    ].copy()

    recommended['similarity_score'] = [
        round(i[1] * 100, 2) for i in filtered_results
    ]

    recommended.insert(0, 'Rank', range(1, len(recommended) + 1))

    return recommended.to_dict(orient="records")