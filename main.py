import numpy as np
import heapq
from google import genai
import os
import pickle
from dotenv import load_dotenv


# 1. Load your .env file
load_dotenv()

# 2. Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(text):
    
    result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
    ) 

    return np.array(result.embeddings[0].values)


def get_top_k(scores, k):
    if k == 0 or k > len(scores): 
        return []
    
    new_scores = heapq.nlargest(k, enumerate(scores), key=lambda x: x[1])
    
    return new_scores 

def safe_normalize(v):
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v/magnitude
    
   

def semantic_search(query, normalize_embeddings, sentences, k):
    
    embedding_query = get_embedding(query)
    normalize_query = safe_normalize(embedding_query)
    
    # Similarity 
    scores = [np.dot(normalize_query, v) for v in normalize_embeddings]

    top_matches = get_top_k(scores, k)
    
    return [(sentences[i], score) for i, score in top_matches]



sentences = [
 "I love machine learning",
 "Python is great for backend",
 "AI is the future",
 "Football is a great sport"
]

CACHE_FILE = r"projects\semantic_search\embeddings.pkl"

# API Call Cost Optimization (Caching)

if os.path.exists(CACHE_FILE):
    print("Loading embeddings from cache...")
    with open(CACHE_FILE, "rb") as f:
        normalized_embeddings = pickle.load(f)
        
else: 
    print("Fetching embeddings from Gemini...")
    raw_embeddings = [get_embedding(v) for v in sentences]
    # Avoid Normalizing Every Query (Pre-normalization)
    normalized_embeddings = [safe_normalize(v) for v in raw_embeddings]
    
    # This will create the folders if they don't exist
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(normalized_embeddings, f)

query = input("Enter your search query: ")
k = int(input("How many top matches you want?: "))
        
results = semantic_search(query, normalized_embeddings, sentences, k)

print(results)

