import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

# -----------------------------
# Path Setup
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "ML", "Models")

# -----------------------------
# Load ML Artifacts
# -----------------------------
tfidf = joblib.load(os.path.join(MODEL_DIR, "recommendation_vectorizer.pkl"))
tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, "recommendation_job_vectors.pkl"))
data = joblib.load(os.path.join(MODEL_DIR, "recommendation_jobs_data.pkl"))

# -----------------------------
# Data Cleaning
# -----------------------------
data = data.fillna("Not Specified")
data['required_experience'] = data['required_experience'].astype(str).str.lower()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# -----------------------------
# Experience Mapping
# -----------------------------
def experience_labels(years):

    try:
        years = int(years)
    except:
        return []

    if years == 0:
        return ["internship", "entry level", "not applicable"]

    elif 1 <= years <= 2:
        return ["associate"]

    elif 3 <= years <= 4:
        return ["executive"]

    else:
        return ["mid-senior"]


# -----------------------------
# Main Recommendation Function
# -----------------------------
def recommend_jobs(user_input, top_n=5, location_filter="", experience_years=None):

    user_input = clean_text(user_input)

    # Get allowed experience labels
    exp_labels = []
    if experience_years is not None:
        exp_labels = experience_labels(experience_years)

    user_vector = tfidf.transform([user_input])

    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[-50:][::-1]

    filtered_results = []

    for idx in top_indices:

        job = data.iloc[idx]

        # Location filter
        if location_filter:
            if location_filter.lower() not in str(job['location']).lower():
                continue

        job_exp = str(job['required_experience']).lower()

        # Experience filtering
        if exp_labels:
            if not any(label in job_exp for label in exp_labels):
                continue

        filtered_results.append((idx, similarity_scores[idx]))

        if len(filtered_results) == top_n:
            break


    # fallback if nothing found
    if not filtered_results:

        for idx in top_indices:

            job = data.iloc[idx]

            if location_filter:
                if location_filter.lower() not in str(job['location']).lower():
                    continue

            filtered_results.append((idx, similarity_scores[idx]))

            if len(filtered_results) == top_n:
                break


    if not filtered_results:
        return []

    # Sort results
    filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

    # Prepare Output
    recommended = data.iloc[[i[0] for i in filtered_results]][
        ['title', 'location', 'industry', 'required_experience']
    ].copy()

    recommended['required_experience'] = recommended['required_experience'].replace("", "Not Specified")
    recommended['industry'] = recommended['industry'].replace("", "Not Specified")
    recommended['location'] = recommended['location'].replace("", "Not Specified")

    recommended.insert(0, 'Rank', range(1, len(recommended) + 1))

    return recommended.to_dict(orient="records")    