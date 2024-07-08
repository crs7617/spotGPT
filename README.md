

---

# SpotGPT: Music Recommendation System


Welcome to **SpotGPT**, the ultimate music recommendation system that leverages the power of Spotify API, Retrieval-Augmented Generation (RAG), and Streamlit to provide you with personalized music recommendations like never before. This isn't just another music recommendation system; it's a sophisticated integration of cutting-edge technologies designed to enhance your music discovery experience.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Introduction

SpotGPT is designed to help you find new music based on your queries. Simply enter the name of any artist and describe the kind of music you're in the mood for, and SpotGPT will do the rest. By harnessing the power of the Spotify API, we fetch real-time data on artists and their tracks, while our RAG system ensures that the recommendations are both relevant and unique.

## Features

- **Artist-Agnostic Recommendations**: Get music recommendations for any artist you like.
- **Spotify Integration**: Uses Spotify API to fetch up-to-date track data.
- **Advanced Natural Language Processing**: Employs Sentence Transformers and GPT-2 for natural and context-aware recommendations.
- **Interactive User Interface**: Built with Streamlit for an engaging and easy-to-use interface.
- **Dynamic Data Retrieval**: Continuously retrieves and updates track information for accurate recommendations.
- **Educational Value**: An excellent project for beginners to understand and implement Retrieval-Augmented Generation (RAG) in a real-world application.

## Tech Stack

- **Programming Language**: Python
- **APIs**: Spotify API
- **Machine Learning**: Sentence Transformers, GPT-2, FAISS
- **Framework**: Streamlit
- **Environment Management**: dotenv

## Installation

Follow these steps to set up SpotGPT on your local machine:

1. **Clone the Repository**
   ```sh
   git clone https://github.com/your-username/spotgpt.git
   cd spotgpt
   ```

2. **Create a Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   - Create a `.env` file in the root directory.
   - Add your Spotify API credentials:
     ```
     SPOTIPY_CLIENT_ID=your_spotify_client_id
     SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
     ```

5. **Run the Application**
   ```sh
   streamlit run main.py
   ```

## Usage

1. **Start the Application**: Run the command `streamlit run main.py`.
2. **Enter Artist Name**: In the input field, type the name of the artist you're interested in.
3. **Submit Your Query**: Provide a description of the type of songs you want to hear.
4. **Get Recommendations**: SpotGPT will fetch and display the recommended tracks based on your input.

## How It Works

### Spotify API Integration

SpotGPT uses the Spotify API to fetch detailed information about artists, albums, and tracks. This ensures that the data is always current and relevant.

### Retrieval-Augmented Generation (RAG)

The core of SpotGPT's recommendation system is based on RAG, which combines retrieval of relevant information with powerful language generation models. Here's a breakdown:

1. **Data Retrieval**: Using the Spotify API, SpotGPT retrieves tracks and their metadata for the specified artist.
2. **Embedding Creation**: Sentence Transformers generate embeddings for each track, capturing the essence of the song's description.
3. **FAISS Indexing**: Embeddings are indexed using FAISS to enable fast and accurate similarity searches.
4. **Contextual Generation**: GPT-2 generates personalized recommendations based on the retrieved tracks and user query.



https://github.com/crs7617/spotGPT/assets/115174268/3c4773b5-dae8-4560-8d6c-51d968dc8e1a



### Streamlit Interface

The entire user experience is crafted using Streamlit, making it easy to interact with SpotGPT and receive recommendations in real-time.

## Contributing

We welcome contributions from the community! If you'd like to contribute to SpotGPT, please follow these steps:

1. **Fork the Repository**
2. **Create a Branch**
   ```sh
   git checkout -b feature/your-feature-name
   ```
3. **Commit Your Changes**
   ```sh
   git commit -m 'Add some feature'
   ```
4. **Push to the Branch**
   ```sh
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

With SpotGPT, dive into the world of music like never before. Discover, listen, and enjoy the tunes that resonate with your unique taste, powered by the synergy of Spotify's vast library and the intelligence of modern NLP models. This project is also an excellent opportunity for beginners to get hands-on experience with RAG and build a practical application using state-of-the-art technologies.


---

