import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from user_based_cf import create_user_item_matrix, user_based_cf, get_user_based_recommendations
from item_based_cf import ItemBasedCF, item_based_cf, get_item_based_recommendations


def load_data():
    cwd = os.getcwd()
    ratings_df = pd.read_csv(cwd + '/datasets/ml-latest-small/ratings.csv')
    movies_df = pd.read_csv(cwd + '/datasets/ml-latest-small/movies.csv')
    return ratings_df, movies_df


def mean_absolute_error(true_ratings, predicted_ratings):
    return np.mean(
        np.abs(
            np.array(true_ratings) - np.array(predicted_ratings)))


def main():
    try:
        print("Loading MovieLens Data")
        ratings_df, movies_df = load_data()
        print(f"Ratings data shape: {ratings_df.shape}")
        print(f"Movies data shape: {movies_df.shape}")

        print("\nCreating User-Item Matrix")
        user_item_matrix = create_user_item_matrix(ratings_df)
        print(f"User-Item Matrix Shape: {user_item_matrix.shape}")

        print("\nInitializing Item-Based CF Model")
        item_cf_model = ItemBasedCF(user_item_matrix)

        print("\nGenerating Recommendations")
        user_id = ratings_df['userId'].min()  # Get the first user ID
        print(f"Generating recommendations for user_id: {user_id}")

        print("\nUser-Based Collaborative Filtering Recommendations:")
        user_based_recommendations = get_user_based_recommendations(
            user_id, user_item_matrix)
        for movie_id, predicted_rating in user_based_recommendations:
            movie_title = movies_df[movies_df['movieId']
                                    == movie_id]['title'].values[0]
            print(f"Movie: {movie_title}, Predicted rating: {predicted_rating:.2f}")

        print("\nItem-Based Collaborative Filtering Recommendations:")
        item_based_recommendations = get_item_based_recommendations(
            user_id, item_cf_model)
        for movie_id, predicted_rating in item_based_recommendations:
            movie_title = movies_df[movies_df['movieId']
                                    == movie_id]['title'].values[0]
            print(f"Movie: {movie_title}, Predicted rating: {predicted_rating:.2f}")

        print("\nEvaluating Recommender Systems")
        train_data, test_data = train_test_split(
            ratings_df, test_size=0.2, random_state=42)
        train_matrix = create_user_item_matrix(train_data)
        train_item_cf_model = ItemBasedCF(train_matrix)

        true_ratings = []
        user_based_predictions = []
        item_based_predictions = []

        for _, row in test_data.iterrows():
            user, movie, true_rating = row['userId'], row['movieId'], row['rating']
            user_based_pred = user_based_cf(user, movie, train_matrix)
            item_based_pred = item_based_cf(user, movie, train_item_cf_model)
            true_ratings.append(true_rating)
            user_based_predictions.append(user_based_pred)
            item_based_predictions.append(item_based_pred)

        user_based_mae = mean_absolute_error(
            true_ratings, user_based_predictions)
        item_based_mae = mean_absolute_error(
            true_ratings, item_based_predictions)

        print(f"User-Based Collaborative Filtering MAE: {user_based_mae:.2f}")
        print(f"Item-Based Collaborative Filtering MAE: {item_based_mae:.2f}")

        sparsity = 1 - \
            (ratings_df.shape[0] / (user_item_matrix.shape[0] * user_item_matrix.shape[1]))
        print(f"\nMatrix Sparsity: {sparsity:.2%}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
