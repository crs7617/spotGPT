import streamlit as st
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Streamlit app title
st.title("SpotGPT: Music Recommendation System")

# Set up Spotify API credentials
os.environ['SPOTIPY_CLIENT_ID'] = os.getenv('SPOTIPY_CLIENT_ID')
os.environ['SPOTIPY_CLIENT_SECRET'] = os.getenv('SPOTIPY_CLIENT_SECRET')

# Initialize Spotify client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

@st.cache_data
def get_artist_albums(artist_name):
    results = sp.search(q=f'artist:{artist_name}', type='artist')
    artist_id = results['artists']['items'][0]['id']
    albums = sp.artist_albums(artist_id, album_type='album')
    album_ids = [album['id'] for album in albums['items']]
    return album_ids

@st.cache_data
def get_album_tracks(album_id):
    results = sp.album_tracks(album_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    
    album_info = sp.album(album_id)
    album_name = album_info['name']
    
    track_data = []
    for track in tracks:
        track_data.append({
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': album_name,
            'popularity': track['popularity'] if 'popularity' in track else None
        })
    return pd.DataFrame(track_data)

# Initialize SentenceTransformer model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('distilbert-base-nli-mean-tokens')

sentence_model = load_sentence_model()

# Function to create embeddings
def create_embedding(row):
    text = f"{row['name']} by {row['artist']} from {row['album']}"
    return sentence_model.encode(text)

# Initialize GPT-2 model and tokenizer
@st.cache_resource
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, gpt2_model

tokenizer, gpt2_model = load_gpt2_model()

# RAG response function
def rag_response(query, k=3):
    # Retrieve
    query_embedding = sentence_model.encode([query])
    D, I = index.search(query_embedding, k)
    retrieved_docs = df.iloc[I[0]]
    
    # Generate
    context = "Retrieved songs:\n"
    for _, row in retrieved_docs.iterrows():
        context += f"{row['name']} by {row['artist']} from {row['album']}\n"
    
    prompt = f"{context}\nQuery: {query}\nResponse:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = gpt2_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# Streamlit UI
st.write("This app recommends songs based on your query.")
artist_name = st.text_input("Enter the artist's name:", "Kanye West")

if st.button("Get Recommendations"):
    with st.spinner("Generating recommendations..."):
        album_ids = get_artist_albums(artist_name)
        all_tracks = pd.DataFrame()
        for album_id in album_ids:
            album_tracks = get_album_tracks(album_id)
            all_tracks = pd.concat([all_tracks, album_tracks], ignore_index=True)
        
        df = all_tracks
        df['embedding'] = df.apply(create_embedding, axis=1)
        
        embeddings = np.array(df['embedding'].tolist())
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        query = st.text_input("Enter your query for song recommendation:", "Recommend a song")
        
        if query:
            response = rag_response(query)
            st.write(response)

        st.subheader("Tracks by " + artist_name)
        st.dataframe(df[['name', 'artist', 'album', 'popularity']])
