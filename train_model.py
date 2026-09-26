"""
Optional offline model-building script.

The Streamlit app can train the TF-IDF matrix automatically on startup,
so running this file is not required for deployment.

Run:
    python train_model.py

This script validates the dataset and reports the number of movies/features.
"""

from recommender import MovieRecommender


if __name__ == "__main__":
    recommender = MovieRecommender(
        "data/tmdb_5000_movies.csv",
        "data/tmdb_5000_credits.csv",
    )

    print("MovieMind model built successfully.")
    print(f"Movies processed: {len(recommender.movies):,}")
    print(f"TF-IDF features: {recommender.tfidf_matrix.shape[1]:,}")
    print("Recommendation engine is ready.")
