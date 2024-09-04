import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def create_user_item_matrix(ratings_df):
    return ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating').fillna(0)


class ItemBasedCF:
    def __init__(self, matrix, k=10):
        self.matrix = matrix
        self.k = k
        self.item_similarities = cosine_similarity(matrix.T)

    def predict(self, user_id, item_id):
        if user_id not in self.matrix.index or item_id not in self.matrix.columns:
            return 0

        item_index = self.matrix.columns.get_loc(item_id)

        # Get similar items (k nearest neighbors)
        similar_items = self.item_similarities[item_index].argsort()[
            ::-1][1:self.k + 1]

        # Predict rating
        user_row = self.matrix.loc[user_id]
        numerator = sum(
            self.item_similarities[item_index][i] *
            user_row.iloc[i] for i in similar_items if not pd.isna(
                user_row.iloc[i]))
        denominator = sum(abs(self.item_similarities[item_index][i])
                          for i in similar_items if not pd.isna(user_row.iloc[i]))

        if denominator == 0:
            return 0
        return numerator / denominator

    def get_recommendations(self, user_id, n=5):
        if user_id not in self.matrix.index:
            return []

        user_row = self.matrix.loc[user_id]
        unrated_items = user_row[user_row == 0].index
        predictions = [(item, self.predict(user_id, item))
                       for item in unrated_items]
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]


def item_based_cf(user_id, item_id, cf_model):
    return cf_model.predict(user_id, item_id)


def get_item_based_recommendations(user_id, cf_model, n=5):
    return cf_model.get_recommendations(user_id, n)
