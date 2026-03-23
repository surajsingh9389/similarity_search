import numpy as np
import heapq
from google import genai
import os
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
   

def semantic_search(query_embedding, sentence_embeddings, sentences, k):
    
    # Normalize 
    normalize_sentence = [v /np.linalg.norm(v) for v in sentence_embeddings]
    normalize_query = query_embedding/np.linalg.norm(query_embedding)
    
    # Similarity 
    scores = [np.dot(normalize_query, v) for v in normalize_sentence]

    top_matches = get_top_k(scores, k)
    
    return [(sentences[i], score) for i, score in top_matches]



sentences = [
 "I love machine learning",
 "Python is great for backend",
 "AI is the future",
 "Football is a great sport"
]

query = "I love Ai"

k = int(input("How many top matches you want?: "))

# Convert all sentences -> embeddings 
sentence_embeddings = [get_embedding(s) for s in sentences]

# Query embedding 
query_embedding = get_embedding(query)
        
top_matches = semantic_search(query_embedding, sentence_embeddings, sentences, k)

print(top_matches)

