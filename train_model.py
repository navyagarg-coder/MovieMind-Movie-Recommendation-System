"""
Optional offline model-building script.

The Streamlit app can train the TF-IDF matrix automatically on startup,
so running this file is not required for deployment.

Run:
    python train_model.py

This script validates the dataset and reports the number of movies/features.
"""

import time

from recommender import MovieRecommender


MOVIES_DATASET_PATH = "data/tmdb_5000_movies.csv"
CREDITS_DATASET_PATH = "data/tmdb_5000_credits.csv"


def main():
    """Build the recommender model and print a validation summary."""
    start_time = time.time()

    recommender = MovieRecommender(
        MOVIES_DATASET_PATH,
        CREDITS_DATASET_PATH,
    )

    elapsed_seconds = time.time() - start_time

    print("MovieMind model built successfully.")
    print(f"Movies processed: {len(recommender.movies):,}")
    print(f"TF-IDF features: {recommender.tfidf_matrix.shape[1]:,}")
    print(f"Build time: {elapsed_seconds:.2f}s")
    print("Recommendation engine is ready.")


if __name__ == "__main__":
    main()