"""
Spotify Dataset Analysis and Recommendation System

This script analyzes a Spotify dataset to generate insights and 
provide a recommendation system based on user preferences for 
artists or playlists.

Dependencies:
- pandas
- numpy
- seaborn
- matplotlib
- mlxtend
- tabulate
"""

# Importing required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from collections import Counter
from tabulate import tabulate
from warnings import filterwarnings

# Suppress specific warnings
filterwarnings("ignore", category=DeprecationWarning)

"""
DATA PREPARATION

The following section involves:
1. Reading the Spotify dataset.
2. Cleaning and processing the data.
3. Exploring the dataset structure and unique features.
"""

# Load the Spotify dataset
spotify_data = pd.read_csv('/kaggle/input/spotify-dataset/spotify_dataset.csv', low_memory=False)

# Inspect the first few rows of the dataset
spotify_data.head()

# Display dataset dimensions
spotify_data.shape

# Inspect unnamed columns for any meaningful data
spotify_data.iloc[:, 5:].head()

# Drop unnecessary columns
spotify_data = spotify_data.drop(spotify_data.columns[4:], axis=1)

# Remove leading whitespaces from column names
spotify_data.columns = [column.strip() for column in spotify_data.columns]
spotify_data.head()

# Limit dataset to the first 50,000 rows
spotify_data = spotify_data.head(50000)

# Check unique entries for various features
spotify_data.user_id.nunique()  # Unique user IDs
spotify_data.artistname.nunique()  # Unique artists
spotify_data.trackname.nunique()  # Unique tracks
spotify_data.playlistname.nunique()  # Unique playlists

# Check for null values in the dataset
spotify_data.isnull().sum().sort_values(ascending=False)

# Value counts for key features
spotify_data["artistname"].value_counts()
spotify_data["trackname"].value_counts()
spotify_data["playlistname"].value_counts()

"""
DATABASE/TRANSACTION ENCODING

Encode the categorical data for 'artistname' and 'playlistname' 
to generate association rules using the Apriori algorithm.

"""

# Encode artist names
min_support = 0.01 
artist_encoded = pd.get_dummies(spotify_data['artistname'])

# Generate association rules for artist names
artist_itemsets = apriori(artist_encoded, min_support=min_support, use_colnames=True)
artist_association_rules = association_rules(artist_itemsets, metric="confidence", min_threshold=0.02, num_itemsets=len(artist_itemsets))

# Encode playlist names
playlist_encoded = pd.get_dummies(spotify_data['playlistname'])

# Generate association rules for playlist names
playlist_itemsets = apriori(playlist_encoded, min_support=min_support, use_colnames=True)
playlist_association_rules = association_rules(playlist_itemsets, metric="confidence", min_threshold=0.02, num_itemsets=len(playlist_itemsets))

"""
VISUALIZATIONS AND INTERPRETATIONS

The following visualizations showcase the top 10 most frequent artists 
and playlists in the dataset.
"""

# Visualize top 10 artists
top_artists = artist_encoded.sum().sort_values(ascending=False).head(10)
plt.bar(top_artists.index, top_artists.values)
plt.xticks(rotation=90)
plt.xlabel('Artist Name')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Artists')
plt.show()

# Visualize top 10 playlists
top_playlists = playlist_encoded.sum().sort_values(ascending=False).head(10)
plt.bar(top_playlists.index, top_playlists.values)
plt.xticks(rotation=90)
plt.xlabel('Playlist Name')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Playlists')
plt.show()

# Analyze strength of association rules
sorted_artist_rules = artist_association_rules.sort_values(by='lift', ascending=False)
sorted_playlist_rules = playlist_association_rules.sort_values(by='lift', ascending=False)

# Display strongest association rules
print("Strongest Association Rules for Artists:")
print(sorted_artist_rules.head())
print("\nStrongest Association Rules for Playlists:")
print(sorted_playlist_rules.head())

"""
PROPOSED RECOMMENDATION SYSTEM

Interactively prompts the user to select preferences (artist or playlist)
and recommends the top tracks based on their selection.
"""

# Prompt the user for input and recommend tracks
choice = input("Enter 'a' to choose by Artist Name or 'p' to choose by Playlist Name: ").upper()

if choice == 'A':
    category = 'artistname'
    category_display = 'Artist Name'
elif choice == 'P':
    category = 'playlistname'
    category_display = 'Playlist Name'
else:
    print("Invalid choice. Please enter 'A' or 'P'.")
    exit()

# Display top 10 choices for the selected category
top_choices = spotify_data[category].value_counts().head(10)
print("\nTop 10 {}:".format(category_display))
for i, choice in enumerate(top_choices.index):
    print("{}. {}".format(i + 1, choice))

# Prompt user to select their preference
selected_index = int(input("\nEnter the number corresponding to your preferred {}: ".format(category_display))) - 1

if selected_index < 0 or selected_index >= len(top_choices):
    print("Invalid input. Please choose a valid number.")
else:
    preferred_category = top_choices.index[selected_index]

    # Filter data based on the user's preferred category
    filtered_songs = spotify_data[spotify_data[category] == preferred_category]

    # Recommend top 10 tracks with associated playlists and artists
    top_tracks = filtered_songs['trackname'].value_counts().head(10).reset_index()
    top_tracks.columns = ['Track Name', 'Frequency']
    top_tracks['Playlist Name'] = filtered_songs.loc[filtered_songs['trackname'].isin(top_tracks['Track Name']), 'playlistname'].unique()[0]
    top_tracks['Artist Name'] = filtered_songs.loc[filtered_songs['trackname'].isin(top_tracks['Track Name']), 'artistname'].unique()[0]

    # Display recommendations in a tabular format
    print("\nTop 10 Track Names, Playlist Names, and Artist Names for {} '{}':".format(category_display, preferred_category))
    print(tabulate(top_tracks[['Track Name', 'Playlist Name', 'Artist Name']], headers='keys', tablefmt='grid'))
