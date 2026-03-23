#####  Semantic Search (Like ChatGPT Retrieval)

# You have sentences:

# sentences = [
#  "I love machine learning",
#  "Python is great for backend",
#  "AI is the future",
#  "Football is a great sport"
# ]

# User query 

# query = "I like AI"


# Goal Find Most similar sentence to query

# -----------------------------------------------------------------


##### Architecture (VERY IMPORTANT)

# sentences
#    ↓
# convert to vectors
#    ↓
# compare with query (cosine similarity)
#    ↓
# rank results
#    ↓
# return best match

