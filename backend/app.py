from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommend_jobs

app = Flask(__name__)
CORS(app)  # Allows frontend to connect

@app.route("/")
def home():
    return "Smart Job Recommendation Backend Running"

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_data = request.json

        skills = user_data.get("skills", "").strip()
        location = user_data.get("location", "").strip()
        experience = user_data.get("experience", "").strip()

        # 🔹 Input Validation
        if not skills:
            return jsonify({"error": "Skills input is required"}), 400

        results = recommend_jobs(
            skills,
            location_filter=location,
            experience_filter=experience
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
