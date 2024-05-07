import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
song_data=pd.read_csv('C:/Users/Lenovo/Downloads/top 100 streamed_songs.csv')
#song_data

st.title('Music Recommendation System')
song_data.isnull().sum()
song_data.duplicated().sum()
song_data=song_data.drop_duplicates()
song_data.duplicated().sum()
#song_data.shape
#song_data.head()
#song_data['danceability']
l=[]
for i in song_data['danceability']:
  l.append(round(i,3))
#l
song_data['danceability']=0.45

cv=CountVectorizer(max_features=2000)
vectors=cv.fit_transform(song_data['name'])

cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)

selected_song = st.selectbox('Select a song', song_data['name'])

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])
def recommend(music):
    song_index = song_data[song_data['name'] == music].index
    if len(song_index) == 0:
        print("Song not found in the dataset.")
        return

    music_index = song_index[0]
    distances = similarity[music_index]
    music_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])
    recommended_songs = []
    for i in range(1, min(6, len(music_list))):
         recommended_songs.append(song_data.iloc[music_list[i][0]]['name'])
    return recommended_songs



if st.button('Recommend'):
    recommended_songs = recommend(selected_song)
    if recommended_songs:
         st.markdown(f"<span style='color: green'>Recommended Songs</span>", unsafe_allow_html=True)
         for song in recommended_songs:
            st.markdown(f"- {song}")
    else:
        st.write("No recommendations found for the selected song.")

#with 🩶,by AK