# music_recommendation_system
A machine learning project that predicts the mood of a song using audio features and recommends similar tracks based on clustering and cosine similarity.

OVERVIEW: This project builds an AI-based music recommendation system using Spotify audio features.It performs two tasks:

(1️)Mood Classification (Using K-Means Clustering):Based on the audio features —
* Danceability
* Energy
* Valence
* Tempo

These are then clustered into these 4 moods:
* 0-	Happy / Positive
* 1	-Sad / Emotional
* 2	-Energetic / Dance
* 3	-Calm / Chill

(2) Song Recommendation (Using Cosine Similarity)

Once the mood cluster is identified, the system:
* Filters songs from the same cluster
* Computes cosine similarity between the new song and all songs in that cluster
*Recommends the top 5 most similar songs

TECH STACK:
*Component	Technology
*Programming	Python
*ML Model	K-Means Clustering
*Feature Scaling	StandardScaler
*Dimensionality Reduction	PCA (for visualization)
*Recommendation Metric Cosine Similarity
*Visuals	Matplotlib

PCA VISUALIZATION:
<img width="671" height="547" alt="image" src="https://github.com/user-attachments/assets/9280d370-26da-420d-b703-6667c2184e09" />

