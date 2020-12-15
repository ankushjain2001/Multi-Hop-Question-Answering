#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Configs
FILENAME_SUFFIX = 'em'
CONTEXT_THRESHOLD = 5 # Number of contexts to keep; Ranges from 2-10;

# --- Get Data ----------------------------------------------------------------

# Import Data
with open('train_'+FILENAME_SUFFIX+'.json') as f:
    train_data = json.load(f)
with open('val_'+FILENAME_SUFFIX+'.json') as f:
    val_data = json.load(f)

# --- Preprocessing -----------------------------------------------------------

def tfidf_preprocessor(question, context):
    vectorizer = TfidfVectorizer(ngram_range=(4, 6), analyzer='char_wb')
    corpus = context.copy()
    corpus.append(question)
    vectorizer.fit(corpus)
    question_vec = vectorizer.transform([question])
    context_vec = vectorizer.transform(context)
    cosine_similarities = linear_kernel(question_vec, context_vec).flatten()
    cosine_similarities = [0.0 if int(x)==1 else x for x in cosine_similarities]
    idx = sorted(range(len(cosine_similarities)), key = lambda x: cosine_similarities[x])[:-5:-1]
    return [context[i] for i in idx]

# Apply query preprocessor
for i in tqdm(train_data):
    if len(i['context']) > CONTEXT_THRESHOLD:
        i['context'] = tfidf_preprocessor(i['question'], i['context'])

for i in tqdm(val_data):
    if len(i['context']) > CONTEXT_THRESHOLD:
        i['context'] = tfidf_preprocessor(i['question'], i['context'])

# --- Save Datasets -----------------------------------------------------------

with open('train_'+FILENAME_SUFFIX+'_pp.json', 'w') as fp:
    json.dump(train_data, fp)
with open('val_'+FILENAME_SUFFIX+'_pp.json', 'w') as fp:
    json.dump(val_data, fp)
