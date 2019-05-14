## Pill-em-all 2.0
<h1> Search and Recommendor system for Medicines </h1>

This is a project developed for searching the the kaggle [data-set](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018/home)

which consists of patient reviews on medications.

**Libraries used**

```
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
stpwrds = stopwords.words("english")
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
import pickle
import time
from itertools import chain
from ast import literal_eval
from sklearn.model_selection import train_test_split
```

**Steps to run the flask app locally**
<body>
<pre>
1.Install flask in your localpython environment. Download the content of this repository into a local folder.

2.Set the FLASK_APP environment variable to app.py and run command **flask run** </b> 
or </b>
Directly run the app.py file by the command **python app.py**

3.Open **localhost:5000** in the web browser to see and interact with the app.
</pre>
</body>

>app.py
**Manages the front end using flask**

>Searchclassify.py

**Creating Inverted Index**


* First implement the vocabulary first part which helps us calculate weights of words.  

```
def Invertedindex(data):
    reviews = data.revvec.apply(literal_eval)
    indices = data.revID.tolist()
    reviews.reindex(indices)
    vocabulary = {}

    for i,r in zip(indices, reviews):
        for j,w in enumerate(r , start=0): ## i is the reviewID, j is the position of word w in the review
            if w not in vocabulary:
                vocabulary[w] = [1,{i:[j]}] ## add review when it isn't in the posting list
            else:
                if i not in vocabulary[w][1]:
                    vocabulary[w][0] += 1    ## count of reviews the word is in
                    vocabulary[w][1][i] = [j] ## append next position of word in the review
                else:
                    vocabulary[w][1][i].append(j) ## append next position of word in the review
                    
  ##.........................................STAGE 1 COMPLETED!!.............................................##
```
* Calculate tf-idf weights for words.

```
    N = np.float64(data.shape[0]) ## total number of reviews
    
    for w in vocabulary.keys():
        pl = {}
        for i in vocabulary[w][1].keys():
            tf = (len(vocabulary[w][1][i])/len(reviews[i]))
            wi = (1 + np.log10(tf)) * np.log10(N/vocabulary[w][0])
            pl[i] = wi
        vocabulary[w].append(pl)
    
  ##.........................................STAGE 2 COMPLETED!!.............................................##

   
```
* Implementing Multinomial Naive Bayes classifier, getting reviewIDs, unique wordsand total word count for each Medical condition(category)

```
def MultiNaiveBayes(data):
    categories = list(set(data.condition))
    train, validation, test = ttv_splitter(data, categories) #custom data splitter
    
    try:
        with open("drugVocabnb.pickle", "rb") as pic:
            vocabulary = pickle.load(pic)
    except:
        start = time.time()
        print("\nCreating vocab for the train data\n")
        vocabulary = Invertedindex(train)             # using the Inverted Index for training dataset
        with open('drugVocabnb.pickle',"wb") as p:
            pickle.dump(vocabulary,p)
        end = time.time()
        print("Done...!", end-start," time\n")
        
    classp = {}
    for c in categories:
        s = train.loc[train['condition'] == c]
        

        reviews = s.revvec.apply(literal_eval)
        uniqwrds = list(set(chain(*reviews)))
        unw = []
                                 
        for w in uniqwrds:
                if w in vocabulary.keys():
                    unw.append(w)
        wc = len(list(chain(*reviews)))
        classp[c] = [ s.revID.tolist(), unw, wc]
    
##.........................................STAGE 1 COMPLETED!!.............................................##
```
* Calculation of weights of each word in a category for every category.
```
    naivecps = {}
    for c in classp.keys():
        naivecps[c] = [len(classp[c][0])/train.shape[0], classp[c][2], len(classp[c][0])] #Conditional probablity of category 
        wrdict = {}
        for w in classp[c][1]:
          lent = 0
          count = 0
          for i in classp[c][0]:
              if i in vocabulary[w][1].keys():
                  lent += len(vocabulary[w][1][i])
                  count += 1
          wrdict[w] = [lent, count]
        naivecps[c].append(wrdict)
      #naivecps[c].append(classp[c][3])
      
##.........................................STAGE 2 COMPLETED!!.............................................##
          
```

>Backend.py

**Preprocessing functions**

```
def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}
    
        return tag_dict.get(tag, wordnet.NOUN)
    
    def rm_stopword(self, r):
        stop_words = ["a", "about", "above", "across", "after", "afterwards","again", "all", "almost", "alone", "along", "already", "also",    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became", "because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between", "beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for","found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they","this", "those", "though", "through", "throughout","thru", "thus", "to", "together", "too", "toward", "towards","under", "until", "up", "upon", "us","very", "was", "we", "well", "were", "what", "whatever", "when","whence", "whenever", "where", "whereafter", "whereas", "whereby","wherein", "whereupon", "wherever", "whether", "which", "while", "who", "whoever", "whom", "whose", "why", "will", "with","within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"]
        r_n = " ".join([i for i in r if i not in stpwrds])
        r_n = " ".join([w for w in r_n.split() if w not in stop_words])
        return r_n
        
    def lem(self, sentence):
        lemmatizer = WordNetLemmatizer()
        out = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)]
        return out
```
* Retrive top k reviews for a given review

```
 def topk(self, query, k=20): #which=0,
        data = self.data
        vocabulary = self.vocabulary
        q = query.replace("[^a-zA-Z]", " ").lower()
        q_vec = self.rm_stopword(q.split())
        q_vect = self.lem(q_vec)
        start = time.time()
        srtdpl = {}
        qw = {}
        allrev = []
        for w in q_vect:
            
            if w in vocabulary.keys():
                if w not in srtdpl.keys():
                    srtdpl[w] = dict(sorted(vocabulary[w][2].items(), key=lambda x:x[1], reverse=True)[:k])
            if w in vocabulary.keys():
                for i in srtdpl[w].keys():
                    if i not in allrev:
                        allrev.append(i)
            if w not in qw:
                qw[w] = [1,(1/len(q_vect))]
            elif w in qw:
                qw[w][0] += 1
                qw[w][1] = (qw[w][0]/len(q_vect))
        if srtdpl == {}:
              return "No results found"
        #elif which == 1:
            #return allrev, q_vect
        #print("\nAllrev", allrev,"\n")
        #print("\n",srtdpl,"\n")
        #print("\n",qw,"\n")
        topk = {}
        sd = 0        
        for d in allrev:
            for w in srtdpl.keys():
                if d in srtdpl[w].keys():
                    sd += srtdpl[w][d] * qw[w][1]
                    
                else:
                    last = list(srtdpl[w].keys())[-1]
                    sd += srtdpl[w][last] * qw[w][1]
            
            topk[d] = sd  
        #print(topk) 
        end = time.time()
        print("Query retrive time:",end - start)
        show = sorted(topk.items(), key=lambda i:i[1], reverse=True)   
        out = []
        for (ind,s) in show:
          out.append(  [data.loc[data.revID == ind, 'revID'].item(), data.loc[data.revID == ind, 'drugName'].item(), data.loc[data.revID == ind, 'usefulCount'].item(), data.loc[data.revID == ind, 'condition'].item(), data.loc[data.revID == ind, 'rating'].item(), data.loc[data.revID == ind, 'review'].item(), s])
          out = out.sort_values(['Similarity','Useful count'], ascending=[False, False])
        
```

 * Classification algorithm that uses Naive bayes priior probablity to predict the probablity of a query belonging to all categories
 
 ```
  def classify(self, review, a=0.0000001):
        naivecps = self.naivecp
        #naivevocablen = self.naivecp['vocablen']
        newr = review.replace("[^a-zA-Z]", " ").lower()
        newr = self.rm_stopword(newr.split())
        new_rev_vec = self.lem(newr)
        
        out = {}
        
        for c in naivecps.keys():
            prc = np.log10(naivecps[c][0]) 
            for w in new_rev_vec:
                if w in naivecps[c][3].keys():
                    prc +=  np.log10((naivecps[c][3][w][0] + a)/(naivecps[c][1] + 32622 ))
                else:
                    prc +=  np.log10((a)/(naivecps[c][1] + 32622 ))
            
            out[c] = prc      
                
        sortout = sorted(out.items(), key=lambda x:x[1], reverse=True)[:10]
        return sortout
```

* Recommendor that takes a reviedID and calls topk() to retrive best amtching reviews and filters the recommendations out

```
 def recommend(self, revID):
        data = self.data
        revID = np.int64(revID)
        c = data.loc[data['revID'] == revID , "condition"].item()
        query = str(data.loc[data['revID'] == revID , "review"].item())
        #print(c, "\n", revID)
        #print("\n\n row:",row,"\n\n")
        #print("\n",query,"\n")
        #simrevID, q_vect = self.topk(query, 1)
        
        #rev_dict = {}
        
       # for ri in simrevID:
           # for w in q_vect:
              #  if self.vocabulary[w][1][ri]:
                  #  rev_dict[ri][w] =  self.vocabulary[w][1][ri][1]
        out = self.topk(query, 50)
       
        if type(out) != str:
            #
            out.sort_values(['Useful count','Condition'], ascending=[False,False], inplace=True)
            out = out.loc[out['Condition'] == c ]
            return out, query
        else:
            return out, query
 ```












