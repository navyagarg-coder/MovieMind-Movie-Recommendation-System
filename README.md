# 🎬 MovieMind — Movie Recommendation System

MovieMind is a content-based movie recommendation system built with Python, Pandas, Scikit-learn and Streamlit.

It recommends movies by comparing their metadata such as:

- Genres
- Keywords
- Plot overview
- Main cast
- Director

## 🚀 How it works

1. Load the TMDB 5000 Movie Dataset.
2. Parse JSON-formatted movie metadata.
3. Combine important movie attributes into one feature representation.
4. Convert the text into numerical vectors using TF-IDF.
5. Calculate similarity using cosine similarity.
6. Return the most similar movies.
7. Display recommendations through a Streamlit web interface.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity
- Streamlit
- Git/GitHub

## 📁 Project Structure

```text
MovieMind/
│
├── app.py
├── recommender.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── tmdb_5000_movies.csv
    └── tmdb_5000_credits.csv
```

## 📊 Dataset

Download the **TMDB 5000 Movie Dataset** from Kaggle.

After downloading, place these two files inside the `data` folder:

```text
tmdb_5000_movies.csv
tmdb_5000_credits.csv
```

Do not rename the files unless you also update the paths in `app.py` and `train_model.py`.

## 💻 Run Locally

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The browser will open the Streamlit application.

## ☁️ Deployment

The project is designed for Streamlit Community Cloud.

1. Push the complete project to GitHub.
2. Make sure `requirements.txt` is in the repository root.
3. Make sure the `data` folder contains both CSV files.
4. Sign in to Streamlit Community Cloud with GitHub.
5. Select the repository and `app.py`.
6. Click Deploy.

## ⚠️ Dataset / API Note

Movie posters are displayed using TMDB image URLs stored through the dataset's `poster_path` values. No TMDB API key is required by this implementation.

## 📌 Future Improvements

- Hybrid collaborative + content-based recommendation
- User accounts and watch history
- Mood-based recommendations
- Genre filters
- Rating-aware ranking
- Search autocomplete
- Recommendation explanations
- Evaluation using Precision@K / Recall@K

## 👩‍💻 Author

Navya Garg
