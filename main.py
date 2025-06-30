# A movie recommendation system using collaborative and content-based filtering,
# trained on 30,000 movies and 1M+ reviews to provide personalized and similar movie suggestions.

import pandas as pd # used for data manipulation and analysis
import numpy as np # used for numerical operations
import joblib # used for saving and loading models
import os # used for file and directory operations
import pickle # used for serializing and deserializing Python objects
from surprise import Dataset, Reader, SVD # used for collaborative filtering
from surprise.model_selection import train_test_split, GridSearchCV # used for splitting data and hyperparameter tuning
from surprise import accuracy # used for evaluating model performance
from sklearn.feature_extraction.text import TfidfVectorizer # used for content-based filtering
from sklearn.neighbors import NearestNeighbors # used for finding similar movies
import seaborn as sns # used for data visualization
import matplotlib.pyplot as plt # used for plotting graphs

# ======================================== Step 1: Load Data ========================================
# Loading the ratings and movies dataframes
# The ratings dataframe contains user_id, movie_id, and rating
ratings_df = pd.read_csv('movie_ratings.csv')  # user_id, movie_id, rating
# The movies dataframe contains movie_id, title, genre, and description
movies_df = pd.read_csv('movies_metadata.csv')  # movie_id, title, genre, description
# The description and genre columns may contain null values, replace them with empty strings
movies_df['description'] = movies_df['description'].fillna('')
# Creating a new column called 'combined' by concatenating the genre and description columns
movies_df['genre'] = movies_df['genre'].fillna('')
# The combined column will be used later for content-based filtering
movies_df['combined'] = movies_df['genre'] + ' ' + movies_df['description']

# Displaying the first few rows of the ratings and movies dataframes
print("\U0001F4DA Movies DataFrame:")
print(movies_df.head())

# ======================================== Step 2: Collaborative Filtering (Surprise) ========================================

# Training Collaborative Filtering Model using SVD with GridSearchCV
# Collaborative Filtering (CF) is a recommendation technique that suggests items
# (e.g., movies, books, products) to a user based on the preferences and behaviors of other similar users.
# In simple terms:
# “People who are similar to you also liked these items.”
print("\u2699\ufe0f  Training Collaborative Filtering Model using SVD with GridSearchCV...")

# 1. Creating a Reader that reads in the ratings dataframe
reader = Reader(rating_scale=(1, 5))

# 2. Loading the data from the ratings dataframe
data = Dataset.load_from_df(ratings_df[['user_id', 'movie_id', 'rating']], reader)

# 3. Spliting the data into a training set and a test set in 80-20 ratio
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 4. Defining hyperparameters to tune using 
# GridSearchCV stands for Grid Search with Cross-Validation.
# It is a technique used in machine learning to automatically find the best combination of hyperparameters for a model.
param_grid = {
    'n_epochs': [15, 20, 25],
    'lr_all': [0.0015, 0.002, 0.0025],
    'reg_all': [0.15, 0.2, 0.25]
}

# 5. Performing GridSearchCV to find the best hyperparameters for the SVD model
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1, joblib_verbose=1)
gs.fit(data)

# 6. Printing the best RMSE score and the best hyperparameters from GridSearchCV
# RMSE stands for Root Mean Square Error.
# It’s a commonly used evaluation metric in regression and recommendation systems,
# to measure the difference between predicted values and actual values.
print(f"✅ Best RMSE Score from GridSearchCV: {gs.best_score['rmse']:.4f}")
print("📌 Best Hyperparameters:", gs.best_params['rmse'])

# 7. Training the SVD model with the best hyperparameters
# SVD: It stands for Singular Value Decomposition, 
# a matrix factorization technique widely used in collaborative filtering to predict user preferences.
cf_model = gs.best_estimator['rmse']
cf_model.fit(trainset)

# 8. Saving the trained SVD model to a file
with open('collaborative_model.pkl', 'wb') as f:
    pickle.dump(cf_model, f)

# 9. Making predictions on the test set using the trained model
predictions = cf_model.test(testset)

# 10. Printing the RMSE score on the test set
print("\n\U0001F4CA Collaborative Filtering RMSE on Test Set:")
accuracy.rmse(predictions)

# Function to get top N recommendations for a user using the SVD model
def get_cf_recommendations(user_id, movie_df, algo, ratings_df, n=5):
    # 1. Getting all the unique movie IDs
    all_movie_ids = movie_df['movie_id'].unique()

    # 2. Getting the movies that the user has seen
    seen = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values

    # 3. Getting the movies that the user has not seen
    unseen = [mid for mid in all_movie_ids if mid not in seen]

    # 4. Making predictions for the user on the unseen movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in unseen]

    # 5. Getting the top N predictions
    top_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # 6. Getting the IDs of the top N movies
    top_ids = [p.iid for p in top_preds]

    # 7. Returning the top N movies
    return movie_df[movie_df['movie_id'].isin(top_ids)][['title', 'genre', 'description']]

# ======================================== Step 3: Content-Based Filtering (TF-IDF) ========================================

# Step 1: Preparing the Content-Based Filtering Model
# Content-Based Filtering (CBF) is a recommendation technique,
# that suggests items to a user based on the attributes of items they've liked before.
print("\n\u2699\ufe0f  Preparing Content-Based Filtering Model...")

# Step 2: Checking if saved models exist
if os.path.exists('tfidf_vectorizer.pkl') and os.path.exists('tfidf_matrix.pkl') and os.path.exists('nearest_neighbors_model.pkl'):
    # Step 3: Loading the existing models if they exist
    print("\U0001F501 Loading saved TF-IDF model and NearestNeighbors model...")
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    nn_model = joblib.load('nearest_neighbors_model.pkl')
else:
    # Step 4: Generating new TF-IDF matrix and train NearestNeighbors model
    # It’s an unsupervised machine learning algorithm used to find items most similar to a given item.
    # I am using this to answer:
    # "What are the most similar movies to this one?"
    print("\U0001F52C Generating TF-IDF matrix and computing similarity...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['combined'])

    # Step 5: Saving the trained TF-IDF vectorizer and matrix
    # TF-IDF stands for Term Frequency–Inverse Document Frequency.
    # It’s a text vectorization method that converts textual data (like movie genres and descriptions) into numerical vectors,
    # so a machine learning model can work with them.
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')

    # Step 6: Training and save NearestNeighbors model
    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)
    joblib.dump(nn_model, 'nearest_neighbors_model.pkl')
    print("✅ NearestNeighbors model trained and saved!")

# Step 7: Creating a series for movie title indices
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

# Step 8: Defining function for content-based recommendations
# Content-Based Recommendations are a type of recommendation system that suggests items (like movies, books, or products),
# based on the features or attributes of the items and the user’s past preferences.
def get_content_recommendations(title, n=5):
    # Step 9: Handling case where title is not found
    if title not in indices:
        return f"Movie '{title}' not found in dataset."
    
    # Step 10: Retrieving the movie index
    idx = indices[title]
    
    # Step 11: Finding the nearest neighbors
    distances, indices_knn = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n+1)
    
    # Step 12: Getting indices of recommended movies
    movie_indices = indices_knn[0][1:]
    
    # Step 13: Returning the recommended movies
    return movies_df.iloc[movie_indices][['title', 'genre', 'description']]

# ======================================== Step 4: Hybrid Recommendations ========================================
# Hybrid Recommendation Systems combine two or more recommendation techniques,
# usually Collaborative Filtering (CF) and Content-Based Filtering (CBF),
# to leverage the strengths of each and minimize their weaknesses.

def get_hybrid_recommendations(user_id, movie_df, cf_model, ratings_df, tfidf_model, tfidf_matrix, nn_model, weight_cf=0.7, weight_cb=0.3, n=5):
    # 1. Getting all the unique movie IDs
    all_movie_ids = movie_df['movie_id'].unique()
    
    # 2. Getting the movies that the user has seen
    seen = ratings_df[ratings_df['user_id'] == user_id]['movie_id'].values
    
    # 3. Getting the movies that the user has not seen
    unseen = [mid for mid in all_movie_ids if mid not in seen]
    
    # 4. Predicting the ratings for unseen movies using the collaborative filtering model
    cf_preds = {movie_id: cf_model.predict(user_id, movie_id).est for movie_id in unseen}
    
    # 5. Getting the top-rated movies by the user
    user_rated = ratings_df[ratings_df['user_id'] == user_id].sort_values(by='rating', ascending=False)
    top_rated_titles = movie_df[movie_df['movie_id'].isin(user_rated['movie_id'])]['title'].head(3).tolist()
    
    # 6. If the user has no rated movies, return an empty DataFrame
    if not top_rated_titles:
        return pd.DataFrame(columns=['title', 'genre', 'description', 'score'])
    
    # 7. Finding similar movies based on the top-rated movies
    similar_movies = set()
    for title in top_rated_titles:
        if title in indices:
            idx = indices[title]
            distances, indices_knn = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=10)
            similar_indices = indices_knn[0][1:]
            similar_movies.update(movies_df.iloc[similar_indices]['movie_id'].values)
    
    # 8. Assigning a content-based score to similar movies that are unseen
    cb_scores = {mid: 1.0 for mid in similar_movies if mid in unseen}
    
    # 9. Calculating hybrid scores for each movie
    hybrid_scores = {}
    for mid in cf_preds:
        cf_score = cf_preds[mid]
        cb_score = cb_scores.get(mid, 0)
        hybrid_scores[mid] = weight_cf * cf_score + weight_cb * cb_score
    
    # 10. Getting the top N movie IDs based on hybrid scores
    top_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:n]
    
    # 11. Creating a result DataFrame with movie details and scores
    result_df = movie_df[movie_df['movie_id'].isin(top_ids)].copy()
    result_df['score'] = result_df['movie_id'].map(hybrid_scores)
    
    # 12. Returning the result DataFrame sorted by score
    return result_df[['title', 'genre', 'description', 'score']].sort_values(by='score', ascending=False)

# ======================================== Step 5: Demo & Export ========================================

# Step 1: Getting the top 5 collaborative filtering recommendations for a user having containing id 123
user_id = 123
print(f"\n\U0001F3AF Top 5 Collaborative Filtering Recommendations for User {user_id}:")
cf_recommendations = get_cf_recommendations(user_id, movies_df, cf_model, ratings_df, n=5)
print(cf_recommendations)

# Step 2: Getting the top 5 content-based recommendations for a movie
example_title = "Arm Discover"
print(f"\n\U0001F3AF Top 5 Content-Based Recommendations similar to '{example_title}':")
content_recommendations = get_content_recommendations(example_title)
print(content_recommendations)

# Step 3: Getting the top 5 hybrid recommendations for a user
print(f"\n\U0001F3AF Top 5 Hybrid Recommendations for User {user_id}:")
hybrid_recommendations = get_hybrid_recommendations(
    user_id=user_id,
    movie_df=movies_df,
    cf_model=cf_model,
    ratings_df=ratings_df,
    tfidf_model=tfidf,
    tfidf_matrix=tfidf_matrix,
    nn_model=nn_model,
    weight_cf=0.7, weight_cb=0.3, n=5
)
print(hybrid_recommendations)

# Step 4: Exporting hybrid recommendations to a CSV file
hybrid_recommendations.to_csv(f"user_{user_id}_hybrid_recommendations.csv", index=False)
print(f"\n\U0001F4E4 Recommendations exported to user_{user_id}_hybrid_recommendations.csv")

# ======================================== Step 6: Visualization ========================================
def plot_genre_distribution(recommended_df):
    # Creating a new figure with a specified size of 10x5 inches
    plt.figure(figsize=(10, 5))
    
    # Counting the occurrences of each genre, and select the top 10
    genre_counts = recommended_df['genre'].value_counts().head(10)
    
    # Plotting a bar chart with genres on the y-axis and their counts on the x-axis
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis")
    
    # Setting the title of the plot
    plt.title("Top Genres in Recommendations")
    
    # Labeling the x-axis
    plt.xlabel("Count")
    
    # Labeling the y-axis
    plt.ylabel("Genre")
    
    # Adjusting the layout to prevent overlap
    plt.tight_layout()
    
    # Displaying the plot
    plt.show()

# Calling the function with the DataFrame containing recommendation data
plot_genre_distribution(hybrid_recommendations)

