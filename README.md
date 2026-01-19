# 🎵 Spotify Songs Genre Segmentation

## 📌 Project Overview
This project focuses on **segmenting Spotify songs into meaningful clusters** using **unsupervised machine learning techniques**.  
By analyzing audio features such as danceability, energy, tempo, and valence, songs are grouped based on musical similarity rather than predefined labels.

The project also studies how these clusters relate to existing **playlist genres** and implements a basic **recommendation system**.

---

## 🎯 Objectives
- Perform **Exploratory Data Analysis (EDA)** on Spotify audio features
- Apply **feature scaling** for distance-based clustering
- Use the **Elbow Method** to determine the optimal number of clusters
- Implement **K-Means Clustering**
- Visualize clusters using **Principal Component Analysis (PCA)**
- Compare clusters with **playlist genres**
- Build a **simple song recommendation system**

---

## 🗂️ Dataset
- **File:** `spotify dataset.csv`
- **Type:** CSV (structured data)

### Audio Features Used
- danceability  
- energy  
- loudness  
- speechiness  
- acousticness  
- instrumentalness  
- liveness  
- valence  
- tempo  

Missing values are handled using **mean imputation**.

---

## 🛠️ Technologies & Libraries
- Python  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
  - StandardScaler  
  - KMeans  
  - PCA  

---

## 🔍 Exploratory Data Analysis
The following visualizations are included:
- Distribution plots for **danceability, energy, and valence**
- Scatter plot: **Energy vs Danceability**
- Box plot: **Loudness across playlist genres**
- **Correlation heatmap** for audio features

---

## 📐 Clustering Methodology

### Feature Scaling
All audio features are standardized using `StandardScaler` since K-Means is distance-based.

### Elbow Method
The **Within-Cluster Sum of Squares (WCSS)** is plotted for different values of K.  
From the elbow curve, **K = 5** is selected as the optimal number of clusters.

### K-Means Clustering
Songs are grouped into **5 clusters** based on audio similarity.

---

## 📊 Visualization
- **PCA (2D)** is used to reduce dimensionality
- Clusters are visualized:
  - By cluster labels
  - By playlist genres
- This helps understand genre overlap and cluster separation

---

## 📋 Cluster vs Playlist Genre
A cross-tabulation table is generated to show the relationship between:
- Cluster assignments
- Playlist genres

This helps evaluate clustering effectiveness.

---

## 🎧 Recommendation System
A simple **cluster-based recommendation system** is implemented.

### Logic:
- Select a song
- Identify its cluster
- Recommend other songs from the same cluster

### Example:
```python
recommend_songs(song_index=0, n=5)
```
Returns recommended track names and their playlist genres.

---
### 🚀 How to Run

1. Clone the repository

2. Place spotify dataset.csv in the same folder

3. Install dependencies:
```
pip install pandas matplotlib seaborn scikit-learn
```

4. Run the script:
```
python spotify_genre_segmentation.py
```
---

### ✅ Key Learnings

- Hands-on experience with unsupervised learning

- Importance of feature scaling

- Cluster interpretation using PCA

- Real-world use of clustering in music recommendation systems

---

### 👨‍🎓 Author

Aayush Raj  
B.Tech CSE (Software Engineering)  
SRM University  
