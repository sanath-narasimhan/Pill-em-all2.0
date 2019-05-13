import numpy as np
import pandas as pd
import pickle
import time
from itertools import chain
from ast import literal_eval
from sklearn.model_selection import train_test_split


def Invertedindex(data):
    reviews = data.revvec.apply(literal_eval)
    indices = data.revID.tolist()
    reviews.reindex(indices)
    vocabulary = {}

    for i,r in zip(indices, reviews):
        for j,w in enumerate(r , start=0):
            if w not in vocabulary:
                vocabulary[w] = [1,{i:[j]}]
            else:
                if i not in vocabulary[w][1]:
                    vocabulary[w][0] += 1
                    vocabulary[w][1][i] = [j]
                else:
                    vocabulary[w][1][i].append(j)
    print("\n\n.........................................STAGE 1 COMPLETED!!.............................................\n\n")
    N = np.float64(data.shape[0])
    
    for w in vocabulary.keys():
        pl = {}
        for i in vocabulary[w][1].keys():
            tf = (len(vocabulary[w][1][i])/len(reviews[i]))
            wi = (1 + np.log10(tf)) * np.log10(N/vocabulary[w][0])
            pl[i] = wi
        vocabulary[w].append(pl)
    
    print("\n\n.........................................STAGE 2 COMPLETED!!.............................................\n\n")

    return vocabulary

def ttv_splitter(data, cat):
    train = pd.DataFrame()
    validation = pd.DataFrame()
    test = pd.DataFrame()

    categories = cat#list(set(y_all))
    

    for c in categories:
      s = data.loc[data['condition'] == c]
      if s.shape[0] < 4 :
        train = train.append(s)
      else:
        tr, intr = train_test_split( s , test_size=0.5, random_state=666 )
        tst, val = train_test_split( intr , test_size=0.5, random_state=777 )
        train = train.append(tr)
        validation = validation.append(val)
        test = test.append(tst)    
    print("data:",data.shape," train:",train.shape," test:",test.shape," validation:", validation.shape )
    return train, validation, test
       
def MultiNaiveBayes(data):
    categories = list(set(data.condition))
    train, validation, test = ttv_splitter(data, categories)
    
    try:
        with open("drugVocabnb.pickle", "rb") as pic:
            vocabulary = pickle.load(pic)
    except:
        start = time.time()
        print("\nCreating vocab for the train data\n")
        vocabulary = Invertedindex(train)
        with open('drugVocabnb.pickle',"wb") as p:
            pickle.dump(vocabulary,p)
        end = time.time()
        print("Done...!", end-start," time\n")
        
    
    ##meds = list(set(data.drugName))
    print("\nLength of classes:", len(categories), "\n")
    classp = {}
    for c in categories:
        s = train.loc[train['condition'] == c]
        
        #mds = s.drugName.tolist()
        reviews = s.revvec.apply(literal_eval)
        uniqwrds = list(set(chain(*reviews)))
        unw = []
        #mdic = {}
        #mdc = list(set(mds))
        #for m in mds:
            #if m not in mdic.keys():
              #  if m in meds:
               #     mdic[m] =  [1, 2/(len(mdc)+len(meds))]
              #  else:
             #       mdic[m] =  [0, 1/(len(mdc)+len(meds))]
            #else:
           #     if m in meds:
          #          mdic[m][0] += 1
         #           mdic[m][1] = (mdic[m][0] + 1)/(len(mdc)+len(meds))
        #    print("\n Done with: ",m,"\n")
                                             
        for w in uniqwrds:
                if w in vocabulary.keys():
                    unw.append(w)
        wc = len(list(chain(*reviews)))
        classp[c] = [ s.revID.tolist(), unw, wc]
    
    print("\n\n.........................................STAGE 1 COMPLETED!!.............................................\n\n")
    naivecps = {}
    for c in classp.keys():
        naivecps[c] = [len(classp[c][0])/train.shape[0], classp[c][2], len(classp[c][0])]
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
      
    print("\n\n.........................................STAGE 2 COMPLETED!!.............................................\n\n", len(vocabulary.keys()))
    #naivecps['vocablen'] = len(vocabulary.keys())
    return  naivecps        

#def Recommend(inrev):
    
    
  