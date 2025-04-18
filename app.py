from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer




app = Flask(__name__)

mylist=[]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus with example sentences


#corpus_embeddings = embedder.encode(string_list, convert_to_tensor=True)

df=pd.read_csv("/home/krupesh/Downloads/gamesreviews/games.csv")


corpus=list(df['Supported languages'])
gamename=list(df['Name'])
gameprice=list(df['Price'])
recommendations=list(df['Recommendations'])
tags=list(df['Tags'])
genres=list(df['Genres'])
website=list(df['Website'])


corpus_embeddings = torch.load('file.pt')




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
   

   top_k = min(30, len(corpus))

   query_embedding = embedder.encode(query, convert_to_tensor=True)

   # We use cosine-similarity and torch.topk to find the highest 5 scores
   similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
   scores, indices = torch.topk(similarity_scores, k=top_k)

   print("\nQuery:", query)
   print("Top 10 most similar sentences in corpus:")

   for score, idx in zip(scores, indices):
      
      gn=gamename[idx]
      gs=corpus[idx]
      gp=gameprice[idx]
      gr=recommendations[idx]
      gg=genres[idx]
      gt=tags[idx]
      iu=website[idx]
      relscoreformatted=f"{score:.4f}"
      results.append({"GameName":gn,"GameInfo":gs,"RelevancyScore":relscoreformatted,"GamePrice":gp,"Recommendations":gr,"Genres":gg,"Tags":gt,"ImageUrl":iu})


   
   
   
      
  
   return results
   





if __name__ == "__main__":
    app.run(debug=True, port=5001)