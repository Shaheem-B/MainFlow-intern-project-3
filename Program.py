import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the dataset into a DataFrame
file_path = "customer_data.xlsx"  # Update this path as needed
df = pd.read_excel(file_path)

# Inspect the dataset
print("Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nNumber of Duplicates:", df.duplicated().sum())
print("\nData Types:\n", df.dtypes)
print("\nSummary Statistics:\n", df.describe())

# Select numerical columns for scaling
num_cols = ["Age", "Annual Income ($)", "Spending Score (1-100)", "Work Experience", "Family Size"]

# Standardize the data using StandardScaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nStandardized Data Sample:\n", df.head(6))

# Determine the optimal number of clusters using the Elbow Method
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df[num_cols])
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Determine the optimal number of clusters using the Silhouette Score
silhouette_scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df[num_cols])
    score = silhouette_score(df[num_cols], cluster_labels)
    silhouette_scores[k] = score

best_k = max(silhouette_scores, key=silhouette_scores.get)
print("Optimal Number of Clusters (Silhouette Score):", best_k)

# Apply K-Means Clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[num_cols])

# 2D Scatter Plot using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df[num_cols])
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df["Cluster"], cmap='Spectral', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Clusters Visualized using PCA')
plt.colorbar(label='Cluster')
plt.show()

# 2D Scatter Plot using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(df[num_cols])
plt.figure(figsize=(8, 6))
plt.scatter(tsne_components[:, 0], tsne_components[:, 1], c=df["Cluster"], cmap='coolwarm', alpha=0.6)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('Clusters Visualized using t-SNE')
plt.colorbar(label='Cluster')
plt.show()

# Pair Plot to visualize relationships
sns.pairplot(df, hue="Cluster", palette="coolwarm", diag_kind="kde")
plt.show()

# Centroid Visuals (Fixed)
centroids = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))

# Plot each centroid
for i, centroid in enumerate(centroids):
    plt.scatter(range(len(num_cols)), centroid, marker='o', label=f'Cluster {i}')

plt.xticks(range(len(num_cols)), num_cols, rotation=45)
plt.title("Cluster Centroids")
plt.ylabel("Standardized Value")
plt.legend()
plt.show()

print("\nClustered Data Sample:\n", df.head())
