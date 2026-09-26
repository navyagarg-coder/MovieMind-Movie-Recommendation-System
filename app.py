import streamlit as st
from recommender import MovieRecommender

st.set_page_config(
    page_title="MovieMind | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 MovieMind")
st.caption("Content-Based Movie Recommendation System using TF-IDF and Cosine Similarity")

@st.cache_resource(show_spinner="Loading recommendation model...")
def load_recommender():
    return MovieRecommender(
        movies_path="data/tmdb_5000_movies.csv",
        credits_path="data/tmdb_5000_credits.csv",
    )

try:
    recommender = load_recommender()
except FileNotFoundError:
    st.error(
        "Dataset files are missing. Add tmdb_5000_movies.csv and "
        "tmdb_5000_credits.csv inside the data/ folder."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the recommendation model: {e}")
    st.stop()

st.sidebar.header("⚙️ Settings")
top_n = st.sidebar.slider("Number of recommendations", 5, 15, 10)

movie_titles = recommender.get_movie_titles()
selected_movie = st.selectbox(
    "🎞️ Choose a movie",
    movie_titles,
    index=0,
)

if st.button("✨ Recommend Movies", type="primary", use_container_width=True):
    recommendations = recommender.recommend(selected_movie, top_n)

    st.subheader(f"Because you selected: **{selected_movie}**")
    st.write("Here are movies with similar content, based on the movie metadata.")

    cols = st.columns(5)

    for i, movie in enumerate(recommendations):
        with cols[i % 5]:
            if movie["poster_url"]:
                st.image(movie["poster_url"], use_container_width=True)
            else:
                st.markdown("### 🎬")
                st.caption("Poster unavailable")

            st.markdown(f"**{movie['title']}**")
            st.caption(
                f"Similarity: {movie['similarity']:.1f}%"
            )

            if movie["genres"]:
                st.caption(f"Genre: {movie['genres']}")

            if movie["overview"]:
                with st.expander("Overview"):
                    st.write(movie["overview"])

st.divider()
st.caption(
    "Dataset: TMDB 5000 Movie Dataset. "
    "Recommendations are generated using content-based filtering."
)
