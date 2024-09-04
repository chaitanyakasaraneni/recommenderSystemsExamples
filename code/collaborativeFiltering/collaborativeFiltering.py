import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Step 1: Load the MovieLens dataset
print("Step 1: Loading MovieLens Data")
cwd = os.getcwd()
ratings_df = pd.read_csv(cwd + '/datasets/ml-latest-small/ratings.csv')
movies_df = pd.read_csv(cwd + '/datasets/ml-latest-small/movies.csv')

print(f"Ratings data shape: {ratings_df.shape}")
print(f"Movies data shape: {movies_df.shape}")

# Step 2: Preprocess the data
print("\nStep 2: Preprocessing Data")
print(ratings_df.head())

# Check for missing values
print("\nMissing values in ratings:")
print(ratings_df.isnull().sum())

# Basic statistics of ratings
print("\nRating statistics:")
print(ratings_df['rating'].describe())

# Step 3: Create user-item matrix
user_item_matrix = ratings_df.pivot(
    index='userId',
    columns='movieId',
    values='rating').fillna(0)
print("\nStep 3: User-Item Matrix Shape")
print(user_item_matrix.shape)

# Step 4: Implement user-based collaborative filtering
def user_based_cf(user_id, item_id, matrix, k=10):
    if user_id not in matrix.index or item_id not in matrix.columns:
        return 0

    user_similarities = cosine_similarity(matrix)
    user_index = matrix.index.get_loc(user_id)

    # Get similar users (k nearest neighbors)
    similar_users = user_similarities[user_index].argsort()[::-1][1:k + 1]

    # Predict rating
    numerator = sum(
        user_similarities[user_index][u] *
        matrix.iloc[u][item_id] for u in similar_users if not pd.isna(
            matrix.iloc[u][item_id]))
    denominator = sum(abs(user_similarities[user_index][u])
                      for u in similar_users if not pd.isna(matrix.iloc[u][item_id]))

    if denominator == 0:
        return 0
    return numerator / denominator

# Step 5: Implement item-based collaborative filtering
def item_based_cf(user_id, item_id, matrix, k=10):
    if user_id not in matrix.index or item_id not in matrix.columns:
        return 0

    item_similarities = cosine_similarity(matrix.T)
    item_index = matrix.columns.get_loc(item_id)

    # Get similar items (k nearest neighbors)
    similar_items = item_similarities[item_index].argsort()[::-1][1:k + 1]

    # Predict rating
    numerator = sum(item_similarities[item_index][i] * matrix.loc[user_id, matrix.columns[i]]
                    for i in similar_items if not pd.isna(matrix.loc[user_id, matrix.columns[i]]))
    denominator = sum(abs(item_similarities[item_index][i]) for i in similar_items if not pd.isna(
        matrix.loc[user_id, matrix.columns[i]]))

    if denominator == 0:
        return 0
    return numerator / denominator

# Step 6: Generate recommendations
def get_recommendations(user_id, matrix, cf_function, n=5):
    if user_id not in matrix.index:
        return []

    user_row = matrix.loc[user_id]
    unrated_items = user_row[user_row == 0].index
    predictions = [(item, cf_function(user_id, item, matrix))
                   for item in unrated_items]
    return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]


# Step 7: Example usage
print("\nStep 7: Generating Recommendations")
user_id = ratings_df['userId'].min()  # Get the first user ID

print("\nUser-Based Collaborative Filtering Recommendations:")
user_based_recommendations = get_recommendations(
    user_id, user_item_matrix, user_based_cf)
for movie_id, predicted_rating in user_based_recommendations:
    movie_title = movies_df[movies_df['movieId']
                            == movie_id]['title'].values[0]
    print(f"Movie: {movie_title}, Predicted rating: {predicted_rating:.2f}")

print("\nItem-Based Collaborative Filtering Recommendations:")
item_based_recommendations = get_recommendations(
    user_id, user_item_matrix, item_based_cf)
for movie_id, predicted_rating in item_based_recommendations:
    movie_title = movies_df[movies_df['movieId']
                            == movie_id]['title'].values[0]
    print(f"Movie: {movie_title}, Predicted rating: {predicted_rating:.2f}")

# Step 8: Evaluation
def mean_absolute_error(true_ratings, predicted_ratings):
    return np.mean(
        np.abs(
            np.array(true_ratings) -
            np.array(predicted_ratings)))


# Split the data into training and testing sets
train_data, test_data = train_test_split(
    ratings_df, test_size=0.2, random_state=42)

# Create user-item matrix for training data
train_matrix = train_data.pivot(
    index='userId',
    columns='movieId',
    values='rating').fillna(0)

true_ratings = []
user_based_predictions = []
item_based_predictions = []

for _, row in test_data.iterrows():
    user, movie, true_rating = row['userId'], row['movieId'], row['rating']
    user_based_pred = user_based_cf(user, movie, train_matrix)
    item_based_pred = item_based_cf(user, movie, train_matrix)
    true_ratings.append(true_rating)
    user_based_predictions.append(user_based_pred)
    item_based_predictions.append(item_based_pred)

user_based_mae = mean_absolute_error(true_ratings, user_based_predictions)
item_based_mae = mean_absolute_error(true_ratings, item_based_predictions)

print(f"\nUser-Based Collaborative Filtering MAE: {user_based_mae:.2f}")
print(f"Item-Based Collaborative Filtering MAE: {item_based_mae:.2f}")

# Step 9: Analyze sparsity
sparsity = 1 - \
    (ratings_df.shape[0] / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
print(f"\nMatrix Sparsity: {sparsity:.2%}")
