# Basic-Movie-Recommendation
🎬 Basic Movie Recommendation
By Mentoga through Skilled Score

A hybrid movie recommendation system built using collaborative filtering and content-based filtering to deliver accurate, diverse, and personalized movie suggestions.

📌 Project Summary
This project processes over 30,000 movies with 1 million+ user reviews to build a hybrid recommendation system. It combines:

Collaborative Filtering (SVD) to learn user preferences based on ratings

Content-Based Filtering (TF-IDF + KNN) to match users with movies having similar genres and descriptions

The final model achieves an RMSE of 1.4222, reflecting strong predictive performance on unseen data.

⚙️ Technologies Used
Data Processing: pandas, numpy, joblib, pickle

Modeling: surprise (SVD, GridSearchCV), scikit-learn (TF-IDF, NearestNeighbors)

Visualization: matplotlib, seaborn

🧠 System Architecture
1. Collaborative Filtering (SVD)
Matrix factorization using the Surprise library

Optimized via GridSearchCV (best RMSE: 1.4229)

Learns latent user-item relationships

2. Content-Based Filtering
TF-IDF vectorization on genres and descriptions

K-Nearest Neighbors with cosine similarity

Identifies similar movies based on content

3. Hybrid Recommendation
Weighted ensemble: 70% Collaborative + 30% Content-Based

Offers a balance between personalization and diversity

📈 Performance
Training RMSE: 1.4229

Testing RMSE: 1.4222

Genre diversity maintained (e.g. Action: 40%, Sci-Fi/Fantasy/Horror: 20% each)

🔍 Sample Hybrid Recommendations (User 123)
Movie Title	Genre	Score
Mission Raise	Action	2.620
Often Whole	Sci-Fi	2.582
Professor Consider	Action	2.545
Worry Upon	Fantasy	2.524
Kitchen Control	Horror	2.512

🚀 Features
Personalized movie recommendations

Content diversity via hybrid design

Model saving for production deployment

CSV export functionality

Scalable and modular pipeline

📂 Project Structure
bash
Copy
Edit
basic-movie-recommendation/
├── data/                     # Ratings and metadata
├── models/                   # Saved model files
├── main.py                   # Run pipeline
├── recommender.ipynb         # Step-by-step notebook
├── utils.py                  # Helper functions
└── README.md                 # Project documentation
🧩 Future Improvements
Integrate metadata (director, cast)

Add implicit feedback (watch time, views)

Implement ranking metrics (precision@k, recall@k)

Use deep learning models for better representations

Deploy with MLFlow, Redis caching, or Spark for scalability


🏅 Internship Acknowledgement
This project was developed during my internship with Mentoga via Skilled Score.

🙋‍♂️ About Me
I'm Adil Shah, an aspiring AI/ML developer from Pakistan 🇵🇰.
I'm passionate about solving real-world problems using data and intelligent systems.
Linkedin Profile: www.linkedin.com/in/syed-adil-shah-8a1537365
GitHub Profile: https://github.com/adil162
