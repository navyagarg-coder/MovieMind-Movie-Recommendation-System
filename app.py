import streamlit as st

from recommender import MovieRecommender


st.set_page_config(
    page_title="MovieMind",
    page_icon="🎬",
    layout="wide"
)


st.title("🎬 MovieMind")

st.subheader(
    "AI-Powered Movie Recommendation System"
)

st.write(
    "Discover movies similar to your favorite movies "
    "using content-based recommendation."
)

st.info(
    "MovieMind analyzes movie genres, keywords, cast, "
    "director, and overview to find similar movies."
)


@st.cache_resource
def load_model():

    return MovieRecommender(
        "data/tmdb_5000_movies.csv",
        "data/tmdb_5000_credits.csv"
    )


try:

    recommender = load_model()

except Exception as error:

    st.error(
        f"Unable to load the recommendation system: {error}"
    )

    st.stop()


movie_titles = recommender.get_movie_titles()


st.sidebar.title("⚙️ MovieMind Settings")

st.sidebar.write(
    "Customize your recommendation experience."
)

number_of_movies = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=15,
    value=10
)

st.sidebar.divider()

st.sidebar.metric(
    "Movies Available",
    len(movie_titles)
)

st.sidebar.caption(
    "Recommendation method: "
    "TF-IDF + Cosine Similarity"
)


selected_movie = st.selectbox(
    "🎞️ Select a movie",
    movie_titles
)


if st.button(
    "✨ Recommend Movies",
    type="primary",
    use_container_width=True
):

    recommendations = recommender.recommend(
        selected_movie,
        number_of_movies
    )

    st.success(
        f"Recommendations based on: {selected_movie}"
    )

    columns = st.columns(5)

    for index, movie in enumerate(
        recommendations
    ):

        with columns[index % 5]:

            if movie["poster_url"]:

                st.image(
                    movie["poster_url"],
                    use_container_width=True
                )

            else:

                st.markdown(
                    "## 🎬"
                )

            st.markdown(
                f"**{movie['title']}**"
            )

            st.caption(
                f"Similarity: "
                f"{movie['similarity']:.1f}%"
            )

            if movie["genres"]:

                st.caption(
                    f"Genre: {movie['genres']}"
                )

            with st.expander(
                "View Overview"
            ):

                if movie["overview"]:

                    st.write(
                        movie["overview"]
                    )

                else:

                    st.write(
                        "Overview not available."
                    )


st.divider()

st.caption(
    "MovieMind uses TF-IDF and cosine similarity "
    "for content-based movie recommendations."
)