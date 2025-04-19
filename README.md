

## EX6 Information Retrieval Using Vector Space Model in Python
# DATE: 19:04:25
## AIM: To implement Information Retrieval Using Vector Space Model in Python.
## Description:
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and sklearn to demonstrate Information Retrieval using the Vector Space Model.
## Procedure:
Define sample documents.
Preprocess text data by tokenizing, removing stopwords, and punctuation.
Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
Execute a sample query and display the search results along with similarity scores.
## Program:
```
NAME : MOHAMED AZEEM N
REG NO : 212222110026

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Sample documents stored in a dictionary
documents = {
"doc1": "Th is is the first document.",
"doc2": "This document is the second document.",
"doc3": "And this is the third one.",
"doc4": "Is this the first document?",
}

# Preprocessing function to tokenize and remove stopwords/punctuation
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words("english") and token not in string.punctuation]
    return " ".join(tokens)
preprocessed_docs = {doc_id: preprocess_text(doc) for doc_id, doc in documents.items()}

# Construct TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_docs.values())

# Calculate cosine similarity between query and documents
def search(query, tfidf_matrix, tfidf_vectorizer):
    preprocessed_query = preprocess_text(query)
    query_vector = tfidf_vectorizer.transform([preprocessed_query])

 # Calculate cosine similarity between query and documents
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)

# Sort documents based on similarity scores
    sorted_indexes = similarity_scores.argsort()[0][::-1]

# Return sorted documents along with their similarity scores
    results = [(list(preprocessed_docs.keys())[i], list(documents.values())[i], similarity_scores[0, i]) for i in sorted_indexes]
    return results

# Get input from user
query = input("Enter your query: ")

# Perform search
search_results = search(query, tfidf_matrix, tfidf_vectorizer)

# Display search results
print("Query:", query)
for i, result in enumerate(search_results, start=1):
    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

# Get the highest rank cosine score
highest_rank_score = max(result[2] for result in search_results)
print("The highest rank cosine score is:", highest_rank_score)
```
## Output:
![Screenshot 2025-04-19 135910](https://github.com/user-attachments/assets/738596a0-30b6-4ee5-8d1b-e0e15b8c2276)

![image](https://github.com/user-attachments/assets/14d85ef6-fa61-4330-adfc-1880b8a94ac4)


## Result:
Execute the code and get the output .
