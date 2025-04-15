from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer



app = Flask(__name__)

mylist=[]

embedder = SentenceTransformer("all-MiniLM-L6-v2")





# file size is too large for github
df=pd.read_html("https://www.kaggle.com/datasets/fronkongames/steam-games-dataset")


corpus=list(df['Supported languages'])
gamename=list(df['Name'])
gameprice=list(df['Price'])
recommendations=list(df['Recommendations'])
tags=list(df['Tags'])
genres=list(df['Genres'])

#Saved model using embeddings from the Supported language column. The model size is too large for github
corpus_embeddings = torch.load('corpusembedding.pt')




# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity





@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form["query"]
        results = mysearch(query)
        return render_template("index.html", results=results, query=query)
    return render_template("index.html", results=[],query="")


def mysearch(query):

   results=[]
   

   top_k = min(20, len(corpus))

   query_embedding = embedder.encode(query, convert_to_tensor=True)

   # We use cosine-similarity and torch.topk to find the highest 5 scores
   similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
   scores, indices = torch.topk(similarity_scores, k=top_k)

   print("\nQuery:", query)
   print("Top 10 most similar sentences in corpus:")

   for score, idx in zip(scores, indices):
      print(idx)
      gn=gamename[idx]
      gs=corpus[idx]
      gp=gameprice[idx]
      gr=recommendations[idx]
      gg=genres[idx]
      gt=tags[idx]
      results.append({"GameName":gn,"GameInfo":gs,"GamePrice":gp,"Recommendations":gr,"Genres":gg,"Tags":gt})


   
   
   
      
  
   return results
   





if __name__ == "__main__":
    app.run(debug=True, port=5001)
