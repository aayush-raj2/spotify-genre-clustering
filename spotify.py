# Spotify Songs Genre Segmentation
# Project 2
# Unsupervised learning using audio features

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# -----------------------------
# Load the dataset
# -----------------------------

# CSV file should be in the same folder
df = pd.read_csv("spotify dataset.csv")

print("Dataset loaded successfully")
print(df.shape)


# -----------------------------
# Data preprocessing
# -----------------------------

# These are the audio features used for clustering
audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

# Fill missing values with mean (simple and effective)
for feature in audio_features:
    df[feature] = df[feature].fillna(df[feature].mean())

# Scale features because KMeans is distance based
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[audio_features])

print("Preprocessing done")


# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Distribution of important features
for feature in ["danceability", "energy", "valence"]:
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.show()

# Relationship between energy and danceability
plt.figure()
sns.scatterplot(x="energy", y="danceability", data=df)
plt.title("Energy vs Danceability")
plt.show()

# Loudness comparison across genres
plt.figure(figsize=(10, 5))
sns.boxplot(x="playlist_genre", y="loudness", data=df)
plt.xticks(rotation=45)
plt.title("Loudness across Playlist Genres")
plt.show()


# -----------------------------
# Correlation matrix
# -----------------------------

plt.figure(figsize=(10, 8))
sns.heatmap(df[audio_features].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Audio Features")
plt.show()


# -----------------------------
# Elbow method to find K
# -----------------------------

wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(scaled_data)
    wcss.append(model.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# -----------------------------
# Apply KMeans clustering
# -----------------------------

# From elbow graph, k = 5 looks reasonable
kmeans = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

print("Clustering completed")
print(df["Cluster"].value_counts())


# -----------------------------
# PCA for visualization
# -----------------------------

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

plt.figure()
plt.scatter(
    pca_result[:, 0],
    pca_result[:, 1],
    c=df["Cluster"]
)
plt.title("Clusters visualized using PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()


# -----------------------------
# Clusters visualized by playlist genre
# (important project requirement)
# -----------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pca_result[:, 0],
    y=pca_result[:, 1],
    hue=df["playlist_genre"],
    style=df["Cluster"],
    s=60
)

plt.title("Song Clusters based on Playlist Genre")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# -----------------------------
# Cluster vs genre table
# -----------------------------

cluster_genre = pd.crosstab(df["Cluster"], df["playlist_genre"])
print("\nCluster vs Playlist Genre")
print(cluster_genre)


# -----------------------------
# Simple recommendation logic
# -----------------------------

def recommend_songs(song_index, n=5):
    """
    Recommends songs from the same cluster
    """
    cluster_id = df.loc[song_index, "Cluster"]
    recommendations = df[df["Cluster"] == cluster_id].sample(n)
    return recommendations[["track_name", "playlist_genre"]]


# Example recommendation
print("\nRecommended songs:")
print(recommend_songs(0))
