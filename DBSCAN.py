"""
DBSCAN Clustering on Yellow Taxi Trip Dataset

This project performs clustering to identify locations with similar trip features
using the DBSCAN algorithm. It also provides analysis on trips belonging to the same
clusters and time-of-day patterns.
"""

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import pyproj

# Data Loading and Preprocessing
"""
Load and preprocess the Yellow Taxi Trip dataset. Handle missing values and prepare
data for clustering.
"""
data = pd.read_parquet('datasets/DBSCAN_data.parquet')
data['passenger_count'].fillna(data['passenger_count'].median(), inplace=True)
data['RatecodeID'].fillna(data['RatecodeID'].mode()[0], inplace=True)
data['store_and_fwd_flag'].fillna(data['store_and_fwd_flag'].mode()[0], inplace=True)
data['congestion_surcharge'].fillna(0, inplace=True)
data['airport_fee'].fillna(0, inplace=True)
data['tpep_pickup_datetime'] = pd.to_datetime(data['tpep_pickup_datetime'])
data['tpep_dropoff_datetime'] = pd.to_datetime(data['tpep_dropoff_datetime'])

# Load Location Data
location_data = pd.read_csv('datasets/DBSCAN_ZoneLookup.csv')

# Subset Data for Clustering
subset_data = data.sample(n=min(100000, len(data)))
subset_data = pd.merge(subset_data, location_data, left_on='PULocationID', right_on='ZoneId')
final_data = subset_data.drop(columns=['VendorID', 'store_and_fwd_flag'])

# DBSCAN Clustering
"""
Apply DBSCAN clustering to identify clusters based on geographic coordinates.
"""
epsilon = 0.01
min_samples = 36
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
final_data['Cluster'] = dbscan.fit_predict(final_data[['Latitude', 'Longitude']])

# Convert Coordinates
proj_in = pyproj.CRS('EPSG:4326')
proj_out = pyproj.CRS('EPSG:4326')
transformer = pyproj.Transformer.from_crs(proj_in, proj_out)
final_data['Latitude'], final_data['Longitude'] = transformer.transform(final_data['Latitude'], final_data['Longitude'])

# GeoDataFrame Creation and Visualization
"""
Create GeoDataFrame and visualize clusters on a map with NYC neighborhood boundaries.
"""
geometry = [Point(xy) for xy in zip(final_data['Longitude'], final_data['Latitude'])]
geo_df = gpd.GeoDataFrame(final_data, geometry=geometry, crs='epsg:4326')
nyc_full = gpd.read_file('datasets/DBSCAN_taxi_zones.zip')
nyc_full = nyc_full.to_crs(epsg=4326)
geo_df = geo_df[geo_df['Cluster'] != -1]

# Plotting
fig, ax = plt.subplots(figsize=(10, 10))
nyc_full.to_crs(epsg=4326).plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', zorder=1)
geo_df.plot(ax=ax, column='Cluster', categorical=True, alpha=0.6, cmap='viridis', markersize=20, linewidth=0.8, zorder=2)
plt.title('Cluster Data Over Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Cluster and Time-of-Day Analysis
"""
Perform analysis on trips belonging to each cluster and generate time-of-day insights.
"""
cluster_analysis = final_data.groupby('Cluster').agg({
    'passenger_count': 'mean',
    'trip_distance': 'mean',
    'fare_amount': 'mean',
    'tip_amount': 'mean',
    'payment_type': lambda x: x.value_counts().index[0],
})
print("Cluster Analysis:\n", cluster_analysis)

final_data['Hour'] = final_data['tpep_pickup_datetime'].dt.hour
hourly_analysis = final_data.groupby(['Cluster', 'Hour']).size().unstack(fill_value=0)

# Heatmap for Temporal-Spatial Analysis
plt.figure(figsize=(12, 8))
sns.heatmap(hourly_analysis, cmap='Blues', linewidths=0.5, linecolor='lightgrey')
plt.title('Temporal-Spatial Distribution of Trips')
plt.xlabel('Hour of Day')
plt.ylabel('Cluster')
plt.show()

# Cost Analysis
"""
Analyze and visualize the average total amount paid by passengers for trips within each cluster.
"""
cluster_avg_cost = final_data.groupby('Cluster')['total_amount'].mean()
plt.figure(figsize=(10, 6))
cluster_avg_cost.plot(kind='bar', color='skyblue')
plt.title('Average Total Amount Paid by Passengers for Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Total Amount ($)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Numerical Feature Analysis
"""
Visualize key numerical features (e.g., passenger count, trip distance) across clusters.
"""
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
for feature in num_features:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=final_data, x='Cluster', y=feature)
    plt.title(f'Mean {feature} Across Clusters')
    plt.xlabel('Cluster')
    plt.ylabel(f'Mean {feature}')
    plt.xticks(rotation=45)
    plt.show()
