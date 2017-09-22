"""
@author: Vipin Chaudhary
@profile: https://github.com/vipin14119
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Visualize:

    def __init__(self, df, sm_mat):
        """
            @param: df -> dataframe of data
            @param: sm_mat -> similarity matrix of tweets
        """

        self.df = df
        self.sm_mat = sm_mat


    def graph(self):
        print "Im drawing a graph of similarity matrix"
