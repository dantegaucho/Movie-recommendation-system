
# Movie Recommendation System

## Table of Contents

- [Overview](#overview)
- [Business Understanding](#business-understanding)
- [Data Understanding](#data-understanding)
- [Data Preparation](#data-preparation)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
 
## Overview

This project implements a personalized movie recommendation engine that suggests the top 5 movies to a user based on their previous ratings. Leveraging machine learning algorithms such as K-Nearest Neighbors (KNN), Singular Value Decomposition (SVD), and Convolutional Neural Networks (CNN), the system is designed for scalability and reproducibility. All code and analysis are provided in Jupyter notebooks for transparency and easy experimentation.

---

## Business Understanding

With the overwhelming number of movies available today, users often struggle to choose what to watch next. Businesses and platforms that provide effective recommendation systems can improve user satisfaction and engagement by personalizing content delivery, ultimately increasing retention and watch time.

*Objectives*
1. To analyze and visualize user preferences and movie trends using the provided ratings and metadata.
2. To implement and evaluate collaborative filtering and matrix factorization models for generating accurate, personalized movie recommendations.
3. To develop a recommendation system that uses past ratings to give personalised movie recommendations.
---

## Data Understanding

- **Dataset:** [MovieLens dataset](https://grouplens.org/datasets/movielens/)
- **Files Used:** `ratings.csv`, `movies.csv`
- **Exploration:** Initial analysis includes exploring user behavior, movie popularity, genre distributions, and sparsity patterns in user ratings.

---

## Data Preparation

- **Cleaning:** Handling missing values and filtering users/movies with insufficient data.
- **Merging:** Combining movie and rating data for unified analysis.
- **Encoding:** Transforming categorical data and scaling features where needed.
- **Splitting:** Partitioning data into training and test sets for modeling.

---

## Modeling

- **K-Nearest Neighbors (KNN):** Memory-based collaborative filtering to find similar users or movies.
- **Singular Value Decomposition (SVD):** Matrix factorization for latent feature extraction using the Surprise library.
- **Convolutional Neural Network (CNN):** Deep learning-based recommender architecture.
- **Hybrid Integration:** Blending models to leverage strengths and improve recommendations.

---

## Evaluation

- **Metrics:** RMSE (Root Mean Squared Error) for prediction accuracy.
- **Validation:** Comparing models using cross-validation and visualizing user-level predictions.
- **Selection:** Choosing the best-performing model (or hybrid) based on accuracy and robustness.

---

## Deployment

- **Notebook Usage:** All steps are reproducible in `notebooks/index.ipynb`.
- **Instructions:**
  1. Download the MovieLens dataset and place `ratings.csv` and `movies.csv` in the `data/` folder.
  2. Install dependencies:
      ```bash
      pip install -r requirements.txt
      ```
  3. Launch the notebook:
      ```bash
      jupyter notebook notebooks/index.ipynb
      ```
  4. Follow the notebook steps: Data Understanding → Preparation → Modeling → Evaluation → Hybrid Integration

---

## Credits

- Daniel Mutiso [Github Link](https://github.com/dantegaucho)
- Teresia Wanjiku [Github Link](https://github.com/tkariuki227)
- Meggy Donna [GitHub](https://github.com/MegAtaro)

Based on the [MovieLens dataset](https://grouplens.org/datasets/movielens/100k/).

---

## License

This project is for educational and academic use. Attribution required if reused.
