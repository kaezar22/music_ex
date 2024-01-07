# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:17:24 2024

@author: ASUS
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Cesar' Music Explorer",
    page_icon=":music:",
    layout="wide",
    )
st.title("Music Explorer")
#Limpieza de Datos

df = pd.read_csv('spotify-2023.csv', encoding = 'latin1')
#pd.set_option('display.max_columns', 1000)
df['streams'] = pd.to_numeric(df['streams'], errors = 'coerce')
df['streams'] = df['streams'].astype('Int64')
df.drop(574, inplace = True)
df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors = 'coerce')
df['in_deezer_playlists'] = df['in_deezer_playlists'].astype('Int64')
df['in_deezer_playlists'].fillna(0, inplace = True)
df.drop('in_shazam_charts', axis = 1, inplace = True)
df['date'] = pd.to_datetime(df[['released_year','released_month','released_day']].astype('str').agg('-'.join, axis = 1), format = '%Y-%m-%d') 

df['key'].fillna('C', inplace = True)
df['key2'] = df['key'] + ' '+ df['mode']
df.drop(['key', 'mode'], axis = 1, inplace = True)
key_encoder = LabelEncoder()
df['key_encoded'] = key_encoder.fit_transform(df['key2'])
key_mapping = dict(zip(df['key_encoded'], df['key2']))


df[['artist1','artist2','artist3']] = df['artist(s)_name'].str.split(', ', expand = True, n = 2)
df.drop('artist(s)_name', axis = 1, inplace = True)
art_encoder = LabelEncoder()
df['art_encoded'] = art_encoder.fit_transform(df['artist1'])
art_mapping = dict(zip(df['artist1'],df['art_encoded']))

def music_explorer():
    
    
    
    #Elegir el año de analisis
    selected_year = st.sidebar.selectbox("Select Year:", df['released_year'].sort_values(ascending = False).unique())
    filtered_df = df[df['released_year'] == selected_year]
    
    #menu de seleccion de artista
    top_artists = filtered_df.groupby('artist1')['streams'].sum().nlargest(10).reset_index()
    select_artist = st.sidebar.selectbox('Select the artist`s top Tracks' , top_artists)
    
    # Using st.columns for layout
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    #Top artist chart
    top_artists2 = top_artists.sort_values(by = 'streams', ascending = False)
    
    # Convert 'artist1' to categorical with specified ordering
    ordering = top_artists2.sort_values(by='streams', ascending=False)['artist1'].tolist()
    top_artists2['artist1'] = pd.Categorical(top_artists2['artist1'], categories=ordering, ordered=True)
    
    col1.subheader("Top Artists of {}".format(selected_year))
    col1.bar_chart(top_artists2.set_index('artist1'), color = '#fcba03')
    
    #top tracks
    top_tracks = filtered_df.sort_values(by = 'streams', ascending = False).head(10)[['track_name','streams']]
    
    # Convert 'track_name' to categorical with specified ordering
    ordering = top_tracks.sort_values(by='streams', ascending=False)['track_name'].tolist()
    top_tracks['track_name'] = pd.Categorical(top_tracks['track_name'], categories=ordering, ordered=True)
    
    col1.subheader("Top Tracks of {}".format(selected_year))
    col3.bar_chart(top_tracks.set_index('track_name'), color = '#0000ff')
    
    #Dataframe with track's authors
    
    
    track_artist = filtered_df.sort_values(by = 'streams', ascending = False).head(10)[['track_name','artist1','artist2']]
    col4.subheader('Top Tracks Authors')
    col4.dataframe(track_artist, height=400)
    
    #number of tracks as maain artist and collaborations
    top_artist3 = filtered_df.groupby(['artist1'])['streams'].sum().sort_values(ascending=False).head(10).index
    df_top_artist3 = df[df['artist1'].isin(top_artist3)]
    
    A = df_top_artist3.groupby('artist1')['track_name'].count().reset_index(name='main_artist')
    df_top_artist4 =df[df['artist2'].isin(top_artist3)]
    B = df_top_artist4.groupby('artist2')['track_name'].count().reset_index(name='colab1')
    df_top_artist5 =df[df['artist3'].isin(top_artist3)]
    C = df_top_artist5.groupby('artist3')['track_name'].count().reset_index(name='colab2')
    
    
    # Merge the DataFrames on the 'artist' column
    merged_df = pd.merge(A, B, left_on='artist1', right_on='artist2', how='outer')
    merged_df = pd.merge(merged_df, C, left_on='artist1', right_on='artist3', how='outer')
    
    merged_df.drop(['artist2','artist3'], axis = 1, inplace = True)
    
    merged_df['colab1'].fillna(0, inplace=True)
    merged_df['colab2'].fillna(0, inplace=True)
    
    # Sum 'colab1' and 'colab2' to get 'colab'
    merged_df['colab'] = merged_df['colab1'] + merged_df['colab2']
    merged_df['total'] = merged_df['main_artist'] + merged_df['colab']
    merged_df.sort_values(by = 'total', ascending = True, inplace = True)
    # Drop 'colab1' and 'colab2' columns if needed
    merged_df.drop(['colab1', 'colab2'], axis=1, inplace=True)
    
    fig = px.bar(merged_df, 
             x=['main_artist', 'colab'], 
             y='artist1', 
             orientation='h',
             color_discrete_map={'main_artist': 'blue', 'colab': 'orange'},
             labels={'main_artist': 'Main Artist', 'colab': 'Collaborations'},
             )

# Display the figure in Streamlit
    col2.subheader("Number of tracks of Top Artists")
    col2.plotly_chart(fig,use_container_width=True)
    
    #choose the artist and check the 10 most popular tracks
    col5, col6 = st.columns([2,1])
    
    # Create a horizontal bar chart
    artist_top_track = df_top_artist3[df_top_artist3['artist1']==select_artist].sort_values(by = 'streams', ascending = True).head(10)
    fig2 = px.bar(
        artist_top_track,
        x='streams',
        y='track_name',
        color='artist1',
        orientation='h',
        
    )
    
    # Display the chart
    col6.subheader('Top Tracks by Selected Artist')
    col6.plotly_chart(fig2, use_container_width=True)
    
    #Line chart of artist evolution
    select_artist2 = st.sidebar.multiselect('Select the artist to compare', top_artists)
    
    filtered_df = df_top_artist3[df_top_artist3['artist1'].isin(select_artist2)]

# Line chart for selected artists using Seaborn
    sns.set(style="darkgrid")
    fig3, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=filtered_df, x='date', y='streams', hue='artist1', marker='o', ax=ax)
    ax.set_title('Streams Over Time for Selected Artists')
    ax.set_xlabel('Date')
    ax.set_ylabel('Streams')
    ax.legend(title='Artist')
    
    # Display the chart
    col5.subheader('Growth of artists')
    col5.write('Use the multi selection box to compare several artists development')
    col5.pyplot(fig3)
    
    # Network
    # Create a directed graph
    fig4, ax2 = plt.subplots(figsize=(12, 6))

    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges based on collaborations
    for index, row in df_top_artist3.iterrows():
        collaborators = [row['artist1'], row['artist2'], row['artist3']]
        collaborators = [artist for artist in collaborators if artist is not None]  # Remove None values
    
        G.add_nodes_from(collaborators)
        for i in range(len(collaborators)):
            for j in range(i + 1, len(collaborators)):
                G.add_edge(collaborators[i], collaborators[j])
    
    # Get positions for each artist in the graph
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
    
    # Draw edges with uniform color
    edges = nx.draw_networkx_edges(G, pos, edge_color=['gray','red','green'], width=2)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Artists')]
    plt.legend(handles=legend_elements, loc='upper left')
    
    # Display the network plot using st.pyplot
    st.subheader('Artist collaboration map of {}'.format(selected_year))
    st.pyplot(fig4)

def music_recommender():
    df_numeric = df[['bpm', 'danceability_%', 'valence_%', 'energy_%',
       'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%','key_encoded','art_encoded']]
    cosine_sim = cosine_similarity(df_numeric, df_numeric)

# Function to get top N similar tracks for a given track name
    def get_top_similar_tracks(track_name, n=10):
        track_index = df[df['track_name'] == track_name].index[0]
        similar_scores = list(enumerate(cosine_sim[track_index]))
        similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)
        
        # Create a DataFrame with track names, artist names, and similarity scores
        similar_tracks_df = pd.DataFrame(columns=["track", "artist", "similarity"])
        for idx, score in similar_scores[1:int(n)+1]:  # Convert num_recommendations to int
            track = df.at[idx, 'track_name']
            artist = df.at[idx, 'artist1']
            similar_tracks_df = pd.concat([similar_tracks_df, pd.DataFrame({"track": [track], "artist": [artist], "similarity": [score]})], ignore_index=True)
    
        return similar_tracks_df
    
    # Streamlit app
    st.title("Music Recommendation System")
    
    # Input box for track name
    track_name2 = st.text_input('Enter the track name:')
    
    # Slider for the number of recommendations
    num_recommendations = st.text_input('Choose the number of recommendations:')
    
    # Button to get recommendations
    if st.button('Get Recommendations'):
        # Get recommendations and display the DataFrame
        top_similar_tracks_df = get_top_similar_tracks(track_name2, num_recommendations)
        st.table(top_similar_tracks_df)
        
def main():
    page = st.radio("Select Page", ["Music Explorer", "Music Recommender"])

    if page == "Music Explorer":
        music_explorer()
    elif page == "Music Recommender":
        music_recommender()
    pass

if __name__ == '__main__':
    main()
 
