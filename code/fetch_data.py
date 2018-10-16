import json
import os

DATA_PATH = '/home/priya/Desktop/IIITD/Semister 1/NLP/Projects/News Compiler/Implemetation/Final_Code/twitter-summarisation/twitter-summarisation/data'

def fetchFiles():
    files = os.listdir(DATA_PATH)
    faulties = []
    data = []
    for txt in files:
        try:
            with open(os.path.join(DATA_PATH, txt), 'r') as raw:
                data.append(json.loads(raw.read()))
        except:
            faulties.append(txt)
    return data
