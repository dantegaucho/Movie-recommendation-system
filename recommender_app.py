import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from surprise import SVD, Reader, Dataset as SurpriseDataset, accuracy, KNNBasic
from sklearn.metrics import precision_score, recall_score, accuracy_score, mean_squared_error

# Collaborative Filtering Recommendation Function
def get_cf_recommendations(user_id, user_item_matrix, movies_df, similar_users, n=10):
    if not len(similar_users):
        return pd.DataFrame()
    if user_id in user_item_matrix.index:
        user_ratings = user_item_matrix.loc[user_id]
        already_rated = set(user_ratings[user_ratings > 0].index)
    else:
        already_rated = set()
    movie_scores = {}
    for _, row in similar_users.iterrows():
        similar_user_id = row['userId']
        similarity = row['similarity']
        similar_user_ratings = user_item_matrix.loc[similar_user_id]
        for movie_id, rating in similar_user_ratings.items():
            if movie_id in already_rated or rating == 0:
                continue
            if movie_id not in movie_scores:
                movie_scores[movie_id] = 0
            movie_scores[movie_id] += rating * similarity
    if not movie_scores:
        return pd.DataFrame()
    cf_recommendations = pd.DataFrame({
        'movieId': list(movie_scores.keys()),
        'cf_score': list(movie_scores.values())
    })
    max_score = cf_recommendations['cf_score'].max()
    if max_score > 0:
        cf_recommendations['cf_score'] = cf_recommendations['cf_score'] / max_score
    cf_recommendations = cf_recommendations.sort_values('cf_score', ascending=False)
    cf_recommendations = cf_recommendations.merge(movies_df[['movieId', 'title', 'genres']], on='movieId')
    return cf_recommendations.head(n)

# SVD Top-N Recommendation Extraction
def get_top_n_recommendations(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if not np.isnan(est):
            if uid not in top_n:
                top_n[uid] = []
            top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def get_recommendations_for_user(user_id, top_n_recommendations, movie_df, top_n=5):
    if user_id in top_n_recommendations:
        recommendations = top_n_recommendations[user_id]
        recommended_movie_ids = [movie_id for movie_id, _ in recommendations]
        recommended_movies = movie_df[movie_df['movieId'].isin(recommended_movie_ids)]
        return recommended_movies[['movieId', 'title', 'genres']]
    else:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

# Hybrid Recommendation Function
def hybrid_recommendation(user_id, user_item_matrix, movie_df, similar_users, top_n_recommendations, top_n=5):
    cf_recs = get_cf_recommendations(
        user_id=user_id,
        user_item_matrix=user_item_matrix,
        movies_df=movie_df,
        similar_users=similar_users,
        n=top_n * 2
    )
    svd_recs = get_recommendations_for_user(user_id, top_n_recommendations, movie_df, top_n=top_n * 2)
    if not cf_recs.empty and not svd_recs.empty:
        cf_ids = set(cf_recs['movieId'])
        svd_ids = set(svd_recs['movieId'])
        both_ids = list(cf_ids & svd_ids)
        only_cf = list(cf_ids - svd_ids)
        only_svd = list(svd_ids - cf_ids)
        hybrid_ids = both_ids + only_cf + only_svd
        hybrid_ids = hybrid_ids[:top_n]
        hybrid_movies = movie_df[movie_df['movieId'].isin(hybrid_ids)][['movieId', 'title', 'genres']]
        return hybrid_movies
    elif not cf_recs.empty:
        return cf_recs[['movieId', 'title', 'genres']].head(top_n)
    elif not svd_recs.empty:
        return svd_recs[['movieId', 'title', 'genres']].head(top_n)
    else:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])

# CNN Dataset and Model
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.user_ids = torch.tensor(df['userId'].values, dtype=torch.long)
        self.movie_ids = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.ratings)
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]

import torch.nn as nn
import torch.optim as optim

class CNNRecommender(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super(CNNRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
        self.conv = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * embedding_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.stack([user_emb, movie_emb], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out.squeeze()

def recommend_cnn(user_id, user_id_map, movie_id_map, movie_ratings_df, model, movie_df, device, top_n=5):
    model.eval()
    user_idx = user_id_map.get(user_id)
    if user_idx is None:
        return pd.DataFrame(columns=['movieId', 'title', 'genres'])
    rated = set(movie_ratings_df[movie_ratings_df['userId'] == user_id]['movieId'])
    candidate_movies = [mid for mid in movie_id_map if mid not in rated]
    candidate_movie_idxs = [movie_id_map[mid] for mid in candidate_movies]
    user_tensor = torch.tensor([user_idx] * len(candidate_movie_idxs), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(candidate_movie_idxs, dtype=torch.long).to(device)
    with torch.no_grad():
        preds = model(user_tensor, movie_tensor).cpu().numpy()
    top_indices = preds.argsort()[-top_n:][::-1]
    top_movie_ids = [candidate_movies[i] for i in top_indices]
    return movie_df[movie_df['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']]

# Model Evaluation Utilities
def evaluate_model(y_true, y_pred, threshold=4.0):
    y_true_bin = [1 if r >= threshold else 0 for r in y_true]
    y_pred_bin = [1 if r >= threshold else 0 for r in y_pred]
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"precision": precision, "recall": recall, "accuracy": accuracy, "rmse": rmse}

# Save and Load SVD Model
def save_svd_model(model, filename='best_svd_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_svd_model(filename='best_svd_model.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)