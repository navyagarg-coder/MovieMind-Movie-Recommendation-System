"""MovieRecommender - content-based movie recommendation engine using TF-IDF."""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

CAST_LIMIT = 5

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


class MovieRecommender:

    def __init__(self, movies_path, credits_path):

        self.movies_path = Path(movies_path)
        self.credits_path = Path(credits_path)

        self.movies = self.load_and_prepare_data()

        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            stop_words="english",
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.movies["combined_features"]
        )

        self.title_to_indices = {}

        for index, title in enumerate(self.movies["title"]):

            key = title.lower()

            if key not in self.title_to_indices:
                self.title_to_indices[key] = []

            self.title_to_indices[key].append(index)

    # -----------------------------
    # JSON PARSER
    # -----------------------------

    @staticmethod
    def parse_json(value):

        if pd.isna(value) or not value:
            return []

        try:
            return json.loads(value)

        except (json.JSONDecodeError, TypeError):

            try:
                return ast.literal_eval(value)

            except (ValueError, SyntaxError):
                return []

    # -----------------------------
    # EXTRACT NAMES
    # -----------------------------

    @classmethod
    def extract_names(cls, value, limit=None):

        data = cls.parse_json(value)

        names = []

        if isinstance(data, list):

            for item in data:

                if isinstance(item, dict):

                    name = item.get("name")

                    if name:
                        names.append(str(name))

        if limit:
            return names[:limit]

        return names

    # -----------------------------
    # EXTRACT CAST
    # -----------------------------

    @classmethod
    def extract_cast(cls, value):

        return cls.extract_names(
            value,
            limit=CAST_LIMIT
        )

    # -----------------------------
    # EXTRACT DIRECTOR
    # -----------------------------

    @classmethod
    def extract_director(cls, value):

        data = cls.parse_json(value)

        if isinstance(data, list):

            for person in data:

                if isinstance(person, dict):

                    if (
                        person.get("job") == "Director"
                        and person.get("name")
                    ):

                        return str(
                            person["name"]
                        )

        return ""

    # -----------------------------
    # CLEAN TEXT
    # -----------------------------

    @staticmethod
    def clean_text(text):

        text = str(text).lower()

        text = re.sub(
            r"[^a-zA-Z0-9\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # -----------------------------
    # LOAD DATA
    # -----------------------------

    def load_and_prepare_data(self):

        if not self.movies_path.exists():

            raise FileNotFoundError(
                f"Movie dataset not found: "
                f"{self.movies_path}"
            )

        if not self.credits_path.exists():

            raise FileNotFoundError(
                f"Credits dataset not found: "
                f"{self.credits_path}"
            )

        movies = pd.read_csv(
            self.movies_path
        )

        credits = pd.read_csv(
            self.credits_path
        )

        # Required columns
        required_movie_columns = {
            "id",
            "title",
            "overview",
            "genres",
            "keywords"
        }

        required_credit_columns = {
            "movie_id",
            "cast",
            "crew"
        }

        missing_movies = (
            required_movie_columns
            - set(movies.columns)
        )

        missing_credits = (
            required_credit_columns
            - set(credits.columns)
        )

        if missing_movies:

            raise ValueError(
                "Movies CSV is missing columns: "
                f"{sorted(missing_movies)}"
            )

        if missing_credits:

            raise ValueError(
                "Credits CSV is missing columns: "
                f"{sorted(missing_credits)}"
            )

        # Convert IDs to numeric
        movies["id"] = pd.to_numeric(
            movies["id"],
            errors="coerce"
        )

        credits["movie_id"] = pd.to_numeric(
            credits["movie_id"],
            errors="coerce"
        )

        # Merge movie and credit datasets
        data = movies.merge(
            credits[
                [
                    "movie_id",
                    "cast",
                    "crew"
                ]
            ],
            left_on="id",
            right_on="movie_id",
            how="left"
        )

        # Remove duplicate movies
        data = data.drop_duplicates(
            subset=["id"]
        ).copy()

        # -----------------------------
        # EXTRACT FEATURES
        # -----------------------------

        data["genres_list"] = data[
            "genres"
        ].apply(
            self.extract_names
        )

        data["keywords_list"] = data[
            "keywords"
        ].apply(
            self.extract_names
        )

        data["cast_list"] = data[
            "cast"
        ].apply(
            self.extract_cast
        )

        data["director"] = data[
            "crew"
        ].apply(
            self.extract_director
        )

        # Convert lists to text
        data["genres_text"] = data[
            "genres_list"
        ].apply(
            lambda x: " ".join(x)
        )

        data["keywords_text"] = data[
            "keywords_list"
        ].apply(
            lambda x: " ".join(x)
        )

        data["cast_text"] = data[
            "cast_list"
        ].apply(
            lambda x: " ".join(x)
        )

        # Handle missing values
        data["overview"] = data[
            "overview"
        ].fillna("")

        data["title"] = data[
            "title"
        ].fillna("Unknown Movie")

        # Poster path is optional
        if "poster_path" not in data.columns:

            data["poster_path"] = ""

        else:

            data["poster_path"] = data[
                "poster_path"
            ].fillna("")

        # -----------------------------
        # COMBINE FEATURES
        # -----------------------------

        data["combined_features"] = (

            data["genres_text"] + " "

            + data["genres_text"] + " "

            + data["keywords_text"] + " "

            + data["keywords_text"] + " "

            + data["cast_text"] + " "

            + data["director"] + " "

            + data["overview"]
        )

        data["combined_features"] = data[
            "combined_features"
        ].apply(
            self.clean_text
        )

        # Remove empty rows
        data = data[
            data["combined_features"].str.len() > 0
        ].reset_index(
            drop=True
        )

        return data

    # -----------------------------
    # GET MOVIE TITLES
    # -----------------------------

    def get_movie_titles(self):

        return self.movies[
            "title"
        ].tolist()

    # -----------------------------
    # FIND MOVIE
    # -----------------------------

    def find_index(self, title):

        key = title.lower().strip()

        matches = self.title_to_indices.get(
            key,
            []
        )

        if not matches:

            raise ValueError(
                f"Movie not found: {title}"
            )

        return matches[0]

    # -----------------------------
    # RECOMMEND MOVIES
    # -----------------------------

    def recommend(
        self,
        title,
        top_n=10
    ):

        index = self.find_index(
            title
        )

        similarity_scores = cosine_similarity(

            self.tfidf_matrix[
                index:index + 1
            ],

            self.tfidf_matrix

        ).flatten()

        # Remove selected movie
        similarity_scores[index] = -1

        sorted_indices = np.argsort(
            similarity_scores
        )[::-1]

        recommendations = []

        selected_title = (
            title.lower().strip()
        )

        seen_titles = {
            selected_title
        }

        for movie_index in sorted_indices:

            movie_title = str(
                self.movies.iloc[
                    movie_index
                ]["title"]
            )

            normalized_title = (
                movie_title.lower().strip()
            )

            if normalized_title in seen_titles:
                continue

            movie = self.movies.iloc[
                movie_index
            ]

            # -----------------------------
            # POSTER
            # -----------------------------

            poster_path = str(
                movie["poster_path"]
            ).strip()

            if poster_path:

                poster_url = (
                    POSTER_BASE_URL
                    + poster_path
                )

            else:

                poster_url = None

            # -----------------------------
            # ADD RECOMMENDATION
            # -----------------------------

            recommendations.append({

                "title": movie_title,

                "similarity": max(
                    0,
                    float(
                        similarity_scores[
                            movie_index
                        ]
                    ) * 100
                ),

                "genres": ", ".join(
                    movie["genres_list"]
                ),

                "overview": str(
                    movie["overview"]
                ),

                "poster_url": poster_url
            })

            seen_titles.add(
                normalized_title
            )

            if len(
                recommendations
            ) >= top_n:

                break

        return recommendations