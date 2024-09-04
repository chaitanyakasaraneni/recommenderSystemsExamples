import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_user_item_matrix(ratings_df):
    return ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating').fillna(0)


def user_based_cf(user_id, item_id, matrix, k=10):
    if user_id not in matrix.index or item_id not in matrix.columns:
        return 0

    user_similarities = cosine_similarity(matrix)
    user_index = matrix.index.get_loc(user_id)

    # Get similar users (k nearest neighbors)
    similar_users = user_similarities[user_index].argsort()[::-1][1:k + 1]

    # Predict rating
    numerator = sum(
        user_similarities[user_index][u] * matrix.iloc[u][item_id] for u in similar_users if not pd.isna(
            matrix.iloc[u][item_id]))
    denominator = sum(abs(user_similarities[user_index][u])
                      for u in similar_users if not pd.isna(matrix.iloc[u][item_id]))

    if denominator == 0:
        return 0
    return numerator / denominator


def get_user_based_recommendations(user_id, matrix, n=5):
    if user_id not in matrix.index:
        return []

    user_row = matrix.loc[user_id]
    unrated_items = user_row[user_row == 0].index
    predictions = [(item, user_based_cf(user_id, item, matrix))
                   for item in unrated_items]
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
