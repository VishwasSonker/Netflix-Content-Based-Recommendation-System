import re
from pathlib import Path
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "netflix_titles.csv"

app = Flask(__name__)

# Load the dataset once when the application starts.
netflix_data = pd.read_csv(DATA_PATH).fillna("")

# Keep original titles for displaying recommendations.
titles = netflix_data["title"].copy()

FEATURES = ["title", "director", "cast", "listed_in", "description"]


def normalize_title(title):
    """Normalize titles so user input is matched reliably."""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def clean_text(value):
    """Lowercase and trim text without removing word boundaries."""
    return str(value).lower().strip()


# Prepare model data.
model_data = netflix_data[FEATURES].copy()

for feature in FEATURES:
    model_data[feature] = model_data[feature].apply(clean_text)


def create_soup(row):
    """Combine the metadata fields used by the recommender."""
    return " ".join(row[FEATURES].tolist())


model_data["soup"] = model_data.apply(create_soup, axis=1)

# TF-IDF gives more importance to informative words than raw word counts.
tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

tfidf_matrix = tfidf.fit_transform(model_data["soup"])

# Build a normalized-title -> row-index lookup.
title_indices = {}

for index, title in model_data["title"].items():
    normalized = normalize_title(title)

    if normalized and normalized not in title_indices:
        title_indices[normalized] = index

# List of titles used by the autocomplete feature.
available_titles = sorted(
    netflix_data["title"].dropna().unique().tolist()
)


def get_recommendations(title, number_of_recommendations=10):
    """
    Return the most similar titles and their similarity percentages.

    Only the selected title is compared against the TF-IDF matrix,
    avoiding a large dense all-pairs cosine-similarity matrix.
    """
    normalized_input = normalize_title(title)

    if not normalized_input:
        return None, "Please enter a movie or TV show name."

    if normalized_input not in title_indices:
        return None, "Movie/TV show not available in the dataset."

    index = title_indices[normalized_input]

    similarity_scores = cosine_similarity(
        tfidf_matrix[index:index + 1],
        tfidf_matrix
    ).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]

    # Remove the selected title itself.
    ranked_indices = [
        i for i in ranked_indices if i != index
    ][:number_of_recommendations]

    recommendations = pd.DataFrame({
        "title": titles.iloc[ranked_indices].values,
        "similarity": similarity_scores[ranked_indices]
    })

    recommendations["similarity"] = (
        recommendations["similarity"] * 100
    ).round(1)

    recommendations.insert(
        0,
        "rank",
        range(1, len(recommendations) + 1)
    )

    return recommendations, None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["GET"])
def search_titles():
    query = request.args.get("q", "").strip().lower()

    if not query:
        return []

    matches = [
        title
        for title in available_titles
        if query in title.lower()
    ]

    return matches[:8]

@app.route("/about", methods=["POST"])
def getvalue():
    movie_name = request.form.get("moviename", "").strip()

    recommendations, error = get_recommendations(movie_name)

    if error:
        return render_template(
            "index.html",
            error=error,
            searched_title=movie_name
        ), 404

    return render_template(
        "result.html",
        recommendations=recommendations.to_dict(orient="records"),
        searched_title=movie_name
    )


if __name__ == "__main__":
    app.run(debug=False)
