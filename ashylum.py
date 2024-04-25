import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io
import numpy as np

# Upload the CSV file
uploaded = files.upload()
filename = next(iter(uploaded))

# Load the dataset
df = pd.read_csv(io.BytesIO(uploaded[filename]))

# Replace '*' with NaN
df = df.replace('*', np.nan)

# Drop rows with NaN values in the columns of interest
df.dropna(subset=['Applied during year', 'Total decisions'], inplace=True)

# Convert columns to numeric
df['Applied during year'] = pd.to_numeric(df['Applied during year'])
df['Total decisions'] = pd.to_numeric(df['Total decisions'])

# Define the features for clustering
X = df[['Applied during year', 'Total decisions']]

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Calculate silhouette score
silhouette_avg = silhouette_score(X, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg:.4f}')

# Create scatter plot for clustering
sns.scatterplot(data=df, x='Applied during year', y='Total decisions', hue='Cluster')
plt.title('Scatter Plot of Applications vs Decisions by Cluster')
plt.show()

# Histogram of applications made during the year
sns.histplot(df['Applied during year'].dropna(), kde=True)
plt.title('Histogram of Applications Made During the Year')
plt.show()

# Line plot for applications over years
sns.lineplot(data=df, x='Year', y='Applied during year', estimator='sum', ci=None)
plt.title('Total Applications Made Over Years')
plt.show()

# Linear regression for 'Total pending end-year' based on 'Applied during year'
X = df[['Applied during year']]
y = df['Total pending end-year']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Plotting regression line over the scatter plot
plt.scatter(X, y, color='blue')
plt.plot(X, regressor.predict(X), color='red', linewidth=2)
plt.title('Regression Line for Total Pending End-Year Cases')
plt.xlabel('Applied during year')
plt.ylabel('Total pending end-year')
plt.show()
