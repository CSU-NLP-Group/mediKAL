import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import json
import re
import pandas as pd
from tqdm import tqdm

input_path = "/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/jingshenke.xlsx"

df = pd.read_excel(input_path)
questions = []
for idx, row in tqdm(df.iterrows(), total = df.shape[0]):
    question = row["病例详情"]
    questions.append(question)

f_in = open("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_based_docs/jingshenke_KG_docs.json", "r")

docs = []

for idx, line in tqdm(enumerate(f_in.readlines()), total = len(f_in.readlines())):
    doc = json.loads(line)["document"]
    docs.append(doc)


sentences = [doc.split() for doc in docs + questions] # 164
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("/home/myjia/Medical_LLM_task/MindMap/data/disease_prediction/kg_based_docs/jingshenke_word2vec.model")
