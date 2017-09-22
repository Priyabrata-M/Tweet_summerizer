"""
@author: Vipin Chaudhary
@profile: https://github.com/vipin14119
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import edit_distance, jaccard_distance


class Similarity:
    def __init__(self, df):
        self.df = df

    def process_data(self):
        print "Common function to process data of twitter"


class TermLevelSimilarity:
    """
        Term level similarity computation class
        @similarities:
            1. Similar URL count
            2. Similar Hashtag count
            3. Similar Username count
            4. Cosine similarity
            5. Jaccard distance
    """

    def __init__(self, df):
        self.df = df
        self.similar_url_count = np.zeros((len(df), len(df)))
        self.similar_ht_count = np.zeros((len(df), len(df)))
        self.similar_un_count = np.zeros((len(df), len(df)))
        self.cosine_similarity = np.zeros((len(df), len(df)))
        self.edit_distance = np.zeros((len(df), len(df)))
        self.jackard_distance = np.zeros((len(df), len(df)))

    def process_data(self):
        for i in range(len(self.df)):
            data = self.df.loc[i]
            text = data.text

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
            self.df.set_value(i, 'text', text)


    def compute_similarity(self):
        for i in range(len(self.df)):
            head = self.df.loc[i]
            head_urls = [ent['url'] for ent in head.entities['urls']]
            head_hashtags = [ent['text'] for ent in head.entities['hashtags']]
            head_usernames = [ent['screen_name'] for ent in head.entities['user_mentions']]

            for j in range(len(self.df)):
                node = self.df.loc[j]
                node_urls = [ent['url'] for ent in node.entities['urls']]
                node_hashtags = [ent['text'] for ent in node.entities['hashtags']]
                node_usernames = [ent['screen_name'] for ent in node.entities['user_mentions']]

                self.similar_url_count[i,j] = len(head_urls) - len(list(set(head_urls) - set(node_urls)))
                self.similar_ht_count[i,j] = len(head_hashtags) - len(list(set(head_hashtags) - set(node_hashtags)))
                self.similar_un_count[i,j] = len(head_usernames) - len(list(set(head_usernames) - set(node_usernames)))
#                 self.edit_distance[i,j] = jaccard_distance(set(head.text.split()), set(node.text.split()))
            print i,

        self.process_data()
        # Compute cosine similarity
        vect = TfidfVectorizer(min_df=1)
        tfidf = vect.fit_transform(self.df.text)
        self.cosine_similarity = (tfidf*tfidf.T).A

        # Compute jackard similarity
        self.add_jackard_distance()
        return self.similar_url_count + self.similar_ht_count + self.similar_un_count + self.cosine_similarity + self.jackard_distance

    def add_jackard_distance(self):

        for i in range(len(self.df)):
            head = self.df.loc[i]
            for j in range(len(self.df)):
                node = self.df.loc[j]
                self.jackard_distance[i,j] = jaccard_distance(set(head.text.split()), set(node.text.split()))

class SemanticLevelSimilarity(Similarity):
    def __init__(self, df):
        self.df = df
        self.sm_mat = np.zeros((len(df), len(df)))

    def process_data(self, df):
        print "Data is being processed"

    def compute_similarity(self, df)
