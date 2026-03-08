
# Install required packages

# pip install gensim

# Import required libraries

from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS

# Step 1: Define corpus
corpus = [
    'king is a strong man', 'queen is a wise woman', 'boy is a young man',
    'girl is a young woman', 'prince is a young', 'prince will be strong',
    'princess is young', 'man is strong', 'woman is pretty',
    'prince is a boy', 'prince will be king', 'princess is a girl',
    'princess will be queen'
]

# Step 2: Tokenize and remove stopwords
statements_listt = [sentence.split() for sentence in corpus]
print("after split statements_listt:",statements_listt)
documents = [[word for word in doc if word not in STOPWORDS] for doc in statements_listt]

# Show the cleaned documents
print("Cleaned Documents:")
print(documents)

# Step 3: Train the Word2Vec model
model = Word2Vec(documents, vector_size=5, window=3, min_count=1)

# Step 4: Vector operations
vector1 = model.wv['king']
vector2 = model.wv['man']
print("\nVector for 'king':", vector1)
print("Vector for man:", vector2)

sum_vector = vector1 + vector2
diff_vector = vector1 - vector2

print("\nSum vector (king + man):", sum_vector)
print("Difference vector (king - man):", diff_vector)

# Step 5: Cosine similarity
similarity = model.wv.similarity('king', 'queen')
print(f"\nCosine Similarity between 'king' and 'queen': {similarity}")

# Step 6: Most similar words
similar_words = model.wv.most_similar('king', topn=5)
print("\nMost Similar words to 'king':", similar_words)

# Step 7: Analogy example
analogy_vector = model.wv['king'] - model.wv['man'] + model.wv['woman']
most_similar = model.wv.most_similar(positive=[analogy_vector], topn=1)
print("\nAnalogy Result (king - man + woman):", most_similar)