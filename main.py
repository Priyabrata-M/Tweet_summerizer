import os
import sys
import pandas as pd
import numpy as np
from gensim import models
import nltk
import random
from nltk import word_tokenize
import matplotlib.pyplot as plt
import json
import re
import time
from sklearn.cluster import DBSCAN
from code.similarities import TermLevelSimilarity
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.cluster import SpectralClustering
stop_words = stopwords.words('english')
from decimal import Decimal


from code.fetch_data import fetchFiles
json_files = fetchFiles()

def get_min_keys(doc):
    min_l = np.Inf
    selected_keys = None
    for json in doc:
        if len(json.keys()) <= min_l:
            min_l = len(json.keys())
            selected_keys = json.keys()
    return selected_keys


   
min_k = [u'contributors', u'truncated', u'text', u'is_quote_status', u'in_reply_to_status_id', u'id', u'favorite_count',
         u'source', u'retweeted', u'coordinates', u'entities', u'in_reply_to_screen_name', u'id_str', u'retweet_count',
         u'in_reply_to_user_id', u'favorited', u'user', u'geo', u'in_reply_to_user_id_str', u'lang', u'created_at',
         u'in_reply_to_status_id_str', u'place', u'metadata']

dict_k = {}
for key in min_k:
    dict_k[key] = []
    
def get_dict(dict_k):
    i = 0
    for files in json_files:
        i += 1
        for json in files:
            for key in min_k:
                dict_k[key].append(json[key])
#         print 'Files ', i    
    return dict_k

root_data = pd.DataFrame(get_dict(dict_k))
# print df.size
df = root_data.loc[0:1000]

vals = df.loc[:10]
for i in range(len(vals)):
    val = vals.loc[i]
#     print val
    print val.text
    print val.entities.keys()
    print val.retweet_count
    for k, v in val.entities.iteritems():
        print 'KEY '+k
        for i in v:
            print i
        print '*'*10
    print '*'*20
    
plt.scatter(x=np.arange(len(df)), y=df.retweet_count)
plt.ylim(0, 100)
plt.ylabel('Number of Retweets')
plt.xlabel('Tweets')
plt.show()





def process_data(frame):
    
    frame = frame.copy()
    for i in range(len(frame)):
        data = frame.loc[i]
        text = data.text
#         print '*'*20
#         print 'ORIGINAL ', text
        # Remove username
        usernames = [ent['screen_name'] for ent in data.entities['user_mentions']] + [data['user']['screen_name']]
        usernames = map(lambda name: '@'+name, usernames)
        big_regex = re.compile('|'.join(map(re.escape, usernames)))
        text = big_regex.sub("", text)
                
        # Just remove Hash
        text = text.replace('#', '')
        
        # Remove urls
        urls = [ent['url'] for ent in data.entities['urls']]
        try:
            urls += [ent['url'] for ent in data.entities['media']]
        except:
            pass
        big_regex = re.compile('|'.join(map(re.escape, urls)))
        text = big_regex.sub("", text)
        
        # Remove Stop words
#         print '*'*20

#         big_regex = re.compile('|'.join(map(re.escape, stop_words)))
#         text = big_regex.sub("", text)
#         print 'NOW ', text
#         print '='*40
        print i,
        frame.set_value(i, 'text', text)
        
    # returning new copy
    return frame
    
# process_data(df)

tic = time.time()
term_level_similarity = TermLevelSimilarity(df)
similarities = term_level_similarity.compute_similarity()
toc = time.time()

print
print 'Time taken to compute term level similarity', (toc - tic)




class SemanticLevelSimilarity:
    '''
        Semantic level similarity computation class
    '''
    def __init__(self, df):
        self.df = df  
  



ls_optimzedclusterCount = [20,25,30]
ls_similarityScore = []
for cluster_count in ls_optimzedclusterCount: 
    sc = SpectralClustering(cluster_count)
    clusters = sc.fit_predict(similarities)
    clustered_tweets = []
    for item in np.unique(clusters):
        temp = df.loc[clusters == item]
        clustered_tweets.append(temp)
    cluster_Score = 0
    for item in clustered_tweets:
        item.index = range(0,len(item))
        term_sim_item = TermLevelSimilarity(item)
        sim_temp = term_sim_item.compute_similarity()
        cluster_Score = cluster_Score + np.sum(sim_temp)
    ls_similarityScore.append(cluster_Score)
    


print "$$$$$$$$$$$$$$$$$$$$$$"
print ls_similarityScore


#for i in range(0,len())   
optimizedNoOfClusters = ls_optimzedclusterCount[ls_similarityScore.index(max(ls_similarityScore))]     
print optimizedNoOfClusters


sc = SpectralClustering(optimizedNoOfClusters)
tic = time.time()
clusters = sc.fit_predict(similarities)
toc = time.time()
print clusters
print len(set(clusters))
print 'Time took to run spectral cluster', (toc-tic)

cluster_dimensions = [clusters[clusters == i].shape[0] for i in range(1, optimizedNoOfClusters+1)]
plt.bar(np.arange(1, optimizedNoOfClusters+1), cluster_dimensions)
plt.xlabel('Cluster number')
plt.ylabel('Number of tweets')
plt.show()

#for n in range(10):
#    db = DBSCAN(min_samples=n+1)
#    db_clusters = db.fit_predict(similarities)
#    num_clus = len(set(db_clusters))
#    print 'Number of Clusters', num_clus, 'for num_samples = ',n+1
#    db_cluster_dimensions = [db_clusters[db_clusters == i].shape[0] for i in range(1, num_clus+1)]
#    plt.bar(np.arange(1, num_clus+1), db_cluster_dimensions)
#    plt.xlabel('Cluster number(min_samples = '+str(n+1)+')')
#    plt.ylabel('Number of tweets')
#    plt.show()
    
#db = DBSCAN(min_samples=20)
#db_clusters = db.fit_predict(similarities)
#num_clus = len(set(db_clusters))
#print 'Number of Clusters', num_clus
#db_cluster_dimensions = [db_clusters[db_clusters == i].shape[0] for i in range(1, num_clus+1)]
#plt.bar(np.arange(1, num_clus+1), db_cluster_dimensions)
#plt.xlabel('Cluster number')
#plt.ylabel('Number of tweets')
#plt.show()
#

def convert_nparray_to_edge(sim_temp):
    edges=[]
    for i in range(0,sim_temp.shape[0]):
        for j in range(0,sim_temp.shape[1]):
            edges.append((i,j,sim_temp[i][j]))
    return edges


clustered_tweets = []
for item in np.unique(clusters):
    temp = df.loc[clusters == item]
    clustered_tweets.append(temp)
    
    
import networkx as nx    
import operator
from code.similarities import TermLevelSimilarity


represented_tweets=[]    
represented_temp=[]
reprentative_text=[]
for item in clustered_tweets:
    item.index = range(0,len(item))
    term_sim_item = TermLevelSimilarity(item)
    sim_temp = term_sim_item.compute_similarity()
    edges = convert_nparray_to_edge(sim_temp)
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    pr = nx.pagerank(G, alpha=0.85,weight='weight')
    sorted_pr = list(item1[0] for item1 in sorted(pr.items(), key=operator.itemgetter(1)))
    represented_tweets.append(item.loc[sorted_pr[0:int(len(sorted_pr)*0.2)]])
    represented_temp.append(sorted_pr)
    reprentative_text.append(item.loc[sorted_pr[0:int(len(sorted_pr)*0.2)]]['text'])
    


RougeFilePtr = open("rougeEval",'w')
for i in range(0,len(reprentative_text)):
    for j in range(0,len(reprentative_text[i])):
        RougeFilePtr.write(reprentative_text[i].tolist()[j].encode("utf-8"))
        RougeFilePtr.write("\n")
RougeFilePtr.close()

RougeFilePtr = open("rougeEval",'r')
strAutoSummary = RougeFilePtr.read()
ls_strAutoSummary = strAutoSummary.split(" ")

ManAnnotatedFilePtr = open("ManuallyAnnoted",'r')
strManAnnotatedSummary = ManAnnotatedFilePtr.read()
ls_ManAnnotatedSummary = strManAnnotatedSummary.split(" ")
ls_bigramManAnnotatedSummary = []
ls_bigramAutoSummary = []
#list all the bigrams in roughfile
for i in range(1,len(ls_strAutoSummary)):
    ls_bigramAutoSummary.append(ls_strAutoSummary[i-1]+" "+ls_strAutoSummary[i])
for i in range(1,len(ls_ManAnnotatedSummary)):
    ls_bigramManAnnotatedSummary.append(ls_ManAnnotatedSummary[i-1]+" "+ls_ManAnnotatedSummary[i])
 
print len(ls_bigramAutoSummary) 
print len(ls_bigramManAnnotatedSummary)
print Decimal(len(list(set(ls_bigramAutoSummary).intersection(ls_bigramManAnnotatedSummary))))*100/Decimal(len(ls_bigramManAnnotatedSummary))
    
    



