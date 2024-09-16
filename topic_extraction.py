# topic_extraction.py

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy

# Download stopwords if not already available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load SpaCy model for advanced NLP
nlp = spacy.load("en_core_web_sm")

# Predefined list of common topics and their associated keywords
predefined_topics = {
    'Product Inquiry': ['product', 'item', 'availability', 'stock', 'details', 'features'],
    'Order Status': ['order', 'status', 'shipment', 'delivery', 'tracking', 'ETA'],
    'Return and Refund': ['return', 'refund', 'exchange', 'policy', 'process', 'refund'],
    # Add more predefined topics as needed
}

def extract_keywords(text, num_keywords=5):
    """
    Extracts the top keywords from the given text using frequency analysis.
    Filters out common stopwords.
    """
    doc = nlp(text.lower())
    words = [token.text for token in doc if token.is_alpha and token.text not in stop_words]
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(num_keywords)
    keywords = [word for word, freq in most_common_words]
    
    return keywords

def extract_topics_sbert(texts, num_clusters=5):
    """
    Extract topics from text using SBERT for embedding and KMeans for clustering.
    Then, extract keywords from each cluster to represent the topics.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller and faster version of BERT
    
    # Generate sentence embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Perform KMeans clustering to group similar sentences
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    
    # Create a dataframe to map each sentence to its cluster
    df = pd.DataFrame({'text': texts, 'cluster': labels})
    
    # Extract the most representative sentence for each cluster
    topics = {}
    for cluster in range(num_clusters):
        cluster_texts = df[df['cluster'] == cluster]['text'].tolist()
        cluster_text = ' '.join(cluster_texts)
        keywords = extract_keywords(cluster_text, num_keywords=5)
        topics[f'Topic {cluster+1}'] = keywords
    
    return topics

def match_topic(keywords, predefined_topics):
    """
    Match the extracted keywords to predefined topics using TF-IDF and cosine similarity.
    """
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    predefined_topic_texts = {topic: ' '.join(keywords) for topic, keywords in predefined_topics.items()}
    
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(keywords) for keywords in predefined_topics.values()] + [' '.join(keywords)])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
    
    best_topic_idx = np.argmax(cosine_similarities)
    best_topic = list(predefined_topics.keys())[best_topic_idx]
    
    return best_topic

def predict_topics_from_transcription(transcription, predefined_topics, num_clusters=1):
    """
    Predict the topic of the transcription, including matched predefined topics and generated new topics.
    """
    transcription_sentences = [sentence.strip() for sentence in transcription.split('.') if sentence.strip()]
    
    # Generate topics using SBERT clustering
    topics = extract_topics_sbert(transcription_sentences, num_clusters=num_clusters)
    
    results = []
    for topic, keywords in topics.items():
        predicted_topic = match_topic(keywords, predefined_topics)
        if predicted_topic:
            results.append(f"Topic: {predicted_topic}")
        else:
            generated_topic = ' '.join(keywords)
            results.append(f"Generated New Topic: {generated_topic}")
    
    return results
