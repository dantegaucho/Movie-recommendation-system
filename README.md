
# Movie Recommendation System
![Banner](LINK)

## Project Overview

This project builds a **personalized movie recommendation system** that suggests the top 5 movies to a user based on their previous ratings. The repository presents a full-featured movie recommendation system that personalizes suggestions based on user ratings. It implements a range of algorithms from basic collaborative filtering to advanced hybrid methods combining traditional ML and deep learning.

The system is designed for scalability and reproducibility, with clear separation between data, models, and visualizations.

---

## Why This Matters

With the ever-growing number of movies, users often find it difficult to decide what to watch. This system uses intelligent algorithms to recommend movies that align with each user's unique preferences.

## Repository Structure

```
.
├── data/                      # Dataset files (download manually)
├── notebooks/
│   └── index.ipynb        # Main notebook with all code and analysis
├── scripts/                   # Optional helper scripts (e.g., for data organization)
├── Images/                    # Visuals used in notebook or presentation
├── requirements.txt           # Python dependencies
├── presentation.pdf           # Final project presentation [link below]
└── README.md                  # This file
```

---

## Data Science Workflow

The project follows these standard data science steps:

1. **Data Collection & Loading** – MovieLens dataset is used.
2. **Exploratory Data Analysis (EDA)** – Visualizations and statistics explore movie trends, genres, and user behavior.
3. **Data Preprocessing** – Cleaning, merging, encoding, and transforming the dataset to prepare it for modeling.
4. **Model Development**:
   - **K-Nearest Neighbors (KNN)** for memory-based collaborative filtering
   - **Singular Value Decomposition (SVD)** via Surprise library
   - **Convolutional Neural Network (CNN)** based deep recommender
5. **Hybrid Model Integration** – Combines the strengths of multiple models to improve accuracy.
6. **Evaluation** – RMSE and user-level predictions guide model comparison and improvement.

All steps are demonstrated in the Jupyter notebook: `index.ipynb`.

---

## Project Links

- PRESENTATION [Presentation (PDF)](LINK)
- DATASET [MovieLens Dataset](LINK)
- NOTEBOOK [Notebook Preview](./PATH)

---

## Navigation Instructions

To explore  this project:

1. **Download Dataset**  
   Visit the [MovieLens website](https://grouplens.org/datasets/movielens/) and download the 100k or 1M dataset. Place `ratings.csv` and `movies.csv` inside the `data/` folder.

2. **Install Dependencies**  
   Create a virtual environment (optional) and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Notebook**  
   Open the notebook with:
   ```bash
   jupyter notebook/VScode notebooks/index.ipynb
   ```

4. **Follow the Steps**  
   Each section is clearly labeled: EDA → Modeling → Evaluation → Hybrid Integration

---

## Credits

- Created by 
1. Daniel Mutiso(GITHUB LINK)
2. Teresia Wanjiku(GITHUB LINK)
3. Meggy Donna(GITHUB LINK)

- Based on the [MovieLens](LINK/) dataset and public ML techniques

---

## License

This project is for educational and academic use. Attribution required if reused.
