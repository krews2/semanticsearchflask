
import torch

from sentence_transformers import SentenceTransformer
import pandas as pd

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus imported

df = pd.read_csv("Insert gaming dataset")

# will use the 'Supported languages' column from the dataset which provides a description of the game which will be used to create the embedding
corpus = list(df['Supported languages'])


# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

#save the embeddings so you can load the model
corpus_embeddings.save('model.pt')

