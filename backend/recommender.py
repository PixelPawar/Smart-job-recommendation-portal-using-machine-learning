import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("../data/processed/legit_jobs.csv")

# Fill missing values
data.fillna("Not Specified", inplace=True)

# Combine important text fields
data['combined_text'] = (
    data['description'] + " " +
    data['requirements'] + " " +
    data['job_text']
)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(data['combined_text'])

def recommend_jobs(user_input, top_n=5, location_filter="", experience_filter=""):

    # Transform user input only (very fast)
    user_vector = tfidf.transform([user_input])

    # Use precomputed matrix (SUPER IMPORTANT)
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)[0]

    # Collect filtered scores
    filtered_scores = []

    for idx, score in enumerate(similarity_scores):

        job = data.iloc[idx]

        if location_filter and location_filter.lower() not in str(job['location']).lower():
            continue

        if experience_filter and experience_filter.lower() not in str(job['required_experience']).lower():
            continue

        filtered_scores.append((idx, score))

    # Sort by similarity
    filtered_scores = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_n]

    recommendations = []

    for idx, score in filtered_scores:
        job = data.iloc[idx]
        recommendations.append({
            "title": job['title'],
            "location": job['location'],
            "industry": job['industry'],
            "experience": job['required_experience']
        })

    return recommendations

