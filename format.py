import json
import pandas as pd
import numpy as np
import gzip
from transformers import pipeline


classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df = getDF('qa_Health_and_Personal_Care.json.gz')


questions_corpus_medical = df['question'][:10000].tolist()
answers_corpus_medical = df['answer'][:10000].tolist()

candidate_labels_health = ['health', 'personal care', 'medical']

data = []
i = 0
for question, answer in zip(questions_corpus_medical, answers_corpus_medical):
    result = classifier(question, candidate_labels=candidate_labels_health)
    best_score = result['scores'][0]
    i += 1
    if best_score > 0.7:
        data.append({"question": question, "answer": answer})
    print(i)
    if i % 1000 == 0:
        with open("health_data.json", "w") as f:
            json.dump(data, f, indent=4)

with open("health_data.json", "w") as f:
    json.dump(data, f, indent=4)