# Netflix Content-Based Recommendation System

A Flask web application that recommends Netflix movies and TV shows based on the content of a title selected by the user.

https://netflix-content-based-recommendatio.vercel.app/

## Features

- Content-based recommendation
- TF-IDF text vectorization
- Cosine similarity
- Top-10 recommendations
- Similarity percentage for each recommendation
- Robust title normalization
- Invalid-title handling
- Responsive Netflix-inspired UI

## How It Works

```text
Netflix Dataset
       |
       v
Missing Value Handling
       |
       v
Text Cleaning
       |
       v
Feature Combination
(title + director + cast + genre + description)
       |
       v
TF-IDF Vectorization
       |
       v
Cosine Similarity
       |
       v
Top 10 Similar Titles
       |
       v
Flask Web Interface
```

## Recommendation Approach

This project uses content-based filtering. Metadata from title, director, cast, genre and description is combined into a text representation. TF-IDF converts that text into vectors and cosine similarity measures similarity between titles.

The application calculates similarity for the selected title at request time instead of storing a dense all-pairs similarity matrix.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Flask
- HTML
- CSS

## Installation

```bash
git clone <your-repository-url>
cd netflix-recommendation-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Dataset

The application uses `netflix_titles.csv` and the following metadata:

- title
- director
- cast
- listed_in
- description

## Limitations

This is a content-based recommender. It does not use user ratings, watch history, user profiles or collaborative filtering. Therefore, the same input title produces the same recommendations for different users.

## Future Improvements

- Movie posters through a metadata API
- Autocomplete search
- Genre/type filters
- Hybrid collaborative + content-based recommendations
- User profiles
