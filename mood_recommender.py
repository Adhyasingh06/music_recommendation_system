import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


df = pd.read_csv("spotify.csv")
print("Dataset loaded. Rows:", len(df)) #to check if all the rows have loaded successfully

# remove duplicate songs based on unique track_id
before = len(df)
df = df.drop_duplicates(subset=['track_id'], keep='first').reset_index(drop=True)
after = len(df)
print(f"Duplicates removed: {before - after}")
print(f"Final rows after cleaning: {after}")


feature_cols = ['danceability', 'energy', 'valence', 'tempo']

df = df.dropna(subset=feature_cols)  # remove missing values
data = df[feature_cols]              # input for clustering


# Standardize Data
scaler = StandardScaler() #creates scaler object
scaled_features = scaler.fit_transform(data)  # transform computes mean and std on data (fit) and returns standardized array.
#standardizing to zero prevents one feature from dominating

k = 4 # number of clusters
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#Asking it to make 4 clusters no training only model
# random_state=42 ensures the result is the same every time you run the code.

clusters = kmeans.fit_predict(scaled_features) #fit and predict training
df['cluster'] = clusters  # assign clusters back into df


print("\nClustering done. Cluster counts:")
print(df['cluster'].value_counts())

# manual mapping from cluster index to a label naming clusters
cluster_names = {
    0: "Happy / Positive",
    1: "Sad / Emotional",
    2: "Energetic / Dance",
    3: "Calm / Chill"
}

df['mood'] = df['cluster'].map(cluster_names) #creates a new mood column


print("\nGenerating PCA cluster visualization...")

pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1],
                      c=df['cluster'], cmap='viridis', alpha=0.6)

plt.title("PCA Visualization of Song Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(scatter, label='Cluster Index')
plt.show()


def predict_mood(song):
    new_song = [[song["danceability"], song["energy"], song["valence"], song["tempo"]]]

    scaled_new = scaler.transform(new_song)

    pred_cluster = kmeans.predict(scaled_new)[0]
    predicted_mood = cluster_names[pred_cluster]

    print("\n Predicted Mood Cluster:", pred_cluster)
    print(" Mood Type:", predicted_mood)

    print("\n Recommended Songs (Cosine Similarity Based):\n")

    # Filter songs from same cluster
    cluster_df = df[df['cluster'] == pred_cluster].reset_index(drop=True)

    # Compute cosine similarity between new song and cluster songs
    cluster_scaled = scaler.transform(cluster_df[feature_cols])
    similarities = cosine_similarity(scaled_new, cluster_scaled)[0]

    # Add similarity scores to dataset
    cluster_df.loc[:, 'similarity'] = similarities  # fixed SettingWithCopyWarning

    # Sort by highest similarity
    best_matches = cluster_df.sort_values(by='similarity', ascending=False).head(5)

    # Print top 5 recommendations
    for _, row in best_matches.iterrows():
        print(f"• {row['track_name']} — {row['artists']} "
              f"[{row.get('track_genre', 'No Genre')}]  | Score: {row['similarity']:.3f}")

    #return pred_cluster, predicted_mood, best_matches


example_song = {
    "danceability": 0.75,
    "energy": 0.80,
    "valence": 0.60,
    "tempo": 120
}

print("\nRunning prediction for example song...")
predict_mood(example_song)
