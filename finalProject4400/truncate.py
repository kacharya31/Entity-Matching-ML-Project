import pandas as pd
import numpy as np
from os.path import join
import csv

csv_file = open("csv_file.csv", "w")
writer = csv.DictWriter(csv_file, fieldnames = ["ltable_id", "rtable_id"])
writer.writeheader()

writer = csv.writer(csv_file)

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('stsb-roberta-large')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

truncate = pd.read_csv('csv_file.csv')
leftTable = pd.read_csv(join("data", "ltable.csv"))
rightTable = pd.read_csv(join("data", "rtable.csv"))

total = len(truncate['ltable_id'])
for i in range(0, 20):
    indexL = int(truncate['ltable_id'][i])
    indexR = int(truncate['rtable_id'][i])
    nameL = leftTable['title'][indexL]
    nameR = rightTable['title'][indexR]
    similarity = cosine(sbert_model.encode([nameL])[0], sbert_model.encode([nameR])[0])
    print("Left: " + nameL + " Right: " + nameR + " Similarity: " + str(similarity))
    if (float(similarity) >= 0.70):
        writer.writerow([truncate['ltable_id'][i], truncate['rtable_id'][i]])
csv_file.close()
