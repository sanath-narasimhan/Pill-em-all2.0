import os
import time
#import pandas as pd
#import json
#import Reviews_to_tdmatrix as tdm
from flask import Flask, render_template, request
#import back_end as be
from Backend import backend as be
 
app = Flask(__name__)
#vocabulary , data = ii.initl()
backend = be()

@app.before_first_request
def tdm_generator():
  #global backend
  #if(os.path.isfile('drugDict.dict') and os.path.isfile('drugCorpus.mm')  == False):
  #  tdm.gen_tdm()
  #backend = be()
  if(os.path.isfile('drugVocab.pickel') == False):
    print(" \n No vocabulary \n ")
  
@app.route('/')
def first():
  return render_template("home.html")
 
@app.route('/home')
def home():
    return render_template("home.html")
    
@app.route('/search', methods = ['POST','GET'])
def search():
    return render_template("index.html")
	
@app.route('/classify', methods = ['POST','GET'])
def classify():
    return render_template("review_index.html")


@app.route('/class_result', methods = ['POST','GET'])
def class_result():
  global backend 
  if request.method == 'POST':
    review = request.form['review']
    start = time.time()
    result = backend.classify(str(review))
    end = time.time()
    print("\n Classification result retrived in ", end-start," \n")
    return render_template("display_class.html", review=review, result=dict(result)) 


@app.route('/result', methods = ['POST','GET'])
def result():
  global backend
  if request.method == 'POST':
    query = request.form['search']
    #simmat = be.similarity(str(query))
    #result = be.topk(simmat)
    start = time.time()
    result = backend.topk(str(query))
    end = time.time()
    if type(result) != str:
      
      print("Query retrive time result:",end - start,"\n") #json.loads(result.to_json(orient='table'))
      return render_template("display_result.html", query=query, result = result.to_dict(orient='records')) #to_html(index=0) #json.loads(result.to_json(orient='columns'))
    else:
      
      print("Query retrive time  no :",end - start)
      return render_template("display_result.html", query=query, result=result)
  
@app.route('/recommend', methods = ['GET'])
def recommend():
    global backend
    if request.method == 'GET':
        revID = int(request.args.get("revID"))
        start = time.time()
        result, query = backend.recommend(revID) 
        #print("\n", query, "\n you clicked on this result!!")
        end = time.time()
        if type(result) != str:
            print("Recommendation retrive time result:",end - start,"\n")
            return render_template("recommend.html",  query=query ,result = result.to_dict(orient='records'))
        else:
            print("Recommendation retrive time no result:",end - start,"\n")
            return render_template("recommend.html", query=query, result=result)
    
  
if __name__ == "__main__":
    app.run()