import os
import json
import pandas as pd
import numpy as np
import nltk
import HTMLParser


class PreProcess:
    '''
        Twitter data pre processing class
    '''
    def __init__(self, df):
        self.df = df.clone()

    def process_data(self):
        self.df.text.apply(lambda text: escapeHtml(text))
        return self.df

    @staticmethod
    def escapeHtml(text):
        html_parser = HTMLParser.HTMLParser()
        return html_parser.unescape(text)
