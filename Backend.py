import time
import numpy as np
import pandas as pd
import pickle
#import pickle
import Searchclassify as scr

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
stpwrds = stopwords.words("english")
from nltk.stem import WordNetLemmatizer


class backend():
    def __init__(self):
        self.data = pd.read_csv("drugData_cleanv1.csv")
        try:
            with open("drugVocabFull.pickle", "rb") as pic:
                self.vocabulary = pickle.load(pic)
        except: 
            start = time.time()
            print("\nCreating vocab for the full data\n")
            self.vocabulary = scr.Invertedindex(self.data)
            with open('drugVocabFull.pickle',"wb") as p:
                pickle.dump(self.vocabulary, p)
            end = time.time()
            print("Done...!", end-start," time\n")
        try:
            with open("drugNaive.pickle", "rb") as cip:
                self.naivecp = pickle.load(cip)
        except:
            start = time.time()
            print("\nCalculating priops for the NBC\n")
            self.naivecp = scr.MultiNaiveBayes(self.data)
            with open('drugNaive.pickle',"wb") as p:
                pickle.dump(self.naivecp, p)
            end = time.time()
            print("Done...!", end-start," time\n")
            
            
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
        
        pd.set_option('display.max_columns', -1)  
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)
       
        out =  pd.DataFrame(out, columns=['ReviewID','Drug Name','Useful count','Condition','Rating(/10)','Review','Similarity'])
        #out = out.sort_values('Useful count', ascending=False)
        return out
    
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
        out = self.topk(query)
       
        if type(out) != str:
            #
            out.sort_values(['Useful count','Condition'], ascending=[False,False], inplace=True)
            out = out.loc[out['Condition'] == c ]
            return out, query
        else:
            return out, query
        
        
        
        
        
                
        
    


    