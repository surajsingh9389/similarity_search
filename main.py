import numpy as np
from google import genai
import os
from dotenv import load_dotenv
import faiss


# 1. Load your .env file
load_dotenv()

# 2. Initialize the client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# convert text to embedding 
def get_embedding(text):
    
    result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text
    ) 

    return np.array(result.embeddings[0].values).astype('float32')

# LLM Response Funtion 
def generate_answer(context, query):
    prompt = f"""
    Answer the question using ONLY the context below.
    
    Context: 
    {context}
    
    Question:
    {query}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    return response.text

# normalize the vector 
def safe_normalize(v):
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    return v/magnitude
    
with open("sentences.txt", "r") as file:
    sentences = [line.strip() for line in file]

# Sentences embedding
raw_embeddings = [get_embedding(v) for v in sentences]

# Sentences Normalization 
normalized_embeddings = [safe_normalize(v) for v in raw_embeddings]

# for extreme speed, float32 type slightly less precision(compare to 64), but uses half the memory. Each number takes 4 bytes.
embeddings_matrix = np.array(normalized_embeddings).astype('float32')

# dimension of embedding
d = embeddings_matrix.shape[1]  # .shape -> (row, cols)
print(d)

# container or database that holds all your vectors 
index = faiss.IndexFlatIP(d) # Inner Product (best for cosine)

# Add embeddigs to the database 
index.add(embeddings_matrix)

query = input("Ask Something: ")
k = 5

# Query Embedding and Normalization 
query_embedding = get_embedding(query)
normalized_query = safe_normalize(query_embedding)
normalized_query = np.array([normalized_query]).astype('float32')

distances, indices = index.search(normalized_query, k)

# Retrieve context 
retrieved_sents = [sentences[i] for i in indices[0]]
                   
context = '\n'.join(retrieved_sents)

# Generate Answer 
answer = generate_answer(context, query)
print(answer)

