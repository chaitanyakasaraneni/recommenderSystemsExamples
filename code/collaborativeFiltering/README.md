# MovieLens Recommender System

This project implements a basic recommender system using the MovieLens Small dataset. It demonstrates both user-based and item-based collaborative filtering techniques.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Customization](#customization)
6. [Limitations](#limitations)

## Prerequisites

To run this code, you need:
- Python 3.7+
- pandas
- numpy
- scikit-learn

## Setup

1. Clone this repository or download the Python script.

2. Install the required Python packages:
   ```
   pip install pandas numpy scikit-learn
   ```

3. Download the MovieLens Small dataset:
   - Go to https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
   - Download and extract the zip file
   - Place the `ratings.csv` and `movies.csv` files in a directory named `ml-latest-small` in the same folder as the Python script

## Code Structure

The project is now split into three main Python files:

1. `user_based_cf.py`: Contains the implementation of user-based collaborative filtering.
2. `item_based_cf.py`: Contains the implementation of item-based collaborative filtering.
3. `main.py`: The main script that uses both collaborative filtering methods to generate recommendations and evaluate their performance.

This modular structure allows for easier understanding and modification of each collaborative filtering method independently.

## Usage

To run the recommender system:

```
python main.py
```

This will execute the main script, which loads the data, generates recommendations using both methods, and evaluates their performance.

The script will:
1. Load and preprocess the MovieLens data
2. Create a user-item matrix
3. Implement user-based and item-based collaborative filtering
4. Generate sample recommendations for a user
5. Evaluate the performance of both methods using Mean Absolute Error
6. Display the sparsity of the user-item matrix

## Code Structure

- Data Loading and Preprocessing: Loads the MovieLens dataset and performs basic data exploration.
- User-Item Matrix Creation: Creates a matrix representation of user ratings for movies.
- Collaborative Filtering Implementation:
  - `user_based_cf`: Implements user-based collaborative filtering
  - `item_based_cf`: Implements item-based collaborative filtering
- Recommendation Generation: `get_recommendations` function generates top N recommendations.
- Evaluation: Calculates Mean Absolute Error for both methods.
- Sparsity Analysis: Calculates the sparsity of the user-item matrix.

## Customization

You can customize the following parameters in the script:
- `k` in `user_based_cf` and `item_based_cf`: Number of nearest neighbors to consider
- `n` in `get_recommendations`: Number of recommendations to generate
- Test set size in `train_test_split`

## Limitations

- This implementation is meant for educational purposes and may not scale well to very large datasets.
- The current version loads the entire dataset into memory, which may not be feasible for larger datasets. Also, you might find this implementation slower.
- Advanced techniques like matrix factorization or hybrid methods are not implemented in this basic version.

Feel free to experiment with the code and extend it for your own projects!