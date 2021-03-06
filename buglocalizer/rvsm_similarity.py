import pickle
import json

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tfidf import TFIDFVectorizer

from datasets import DATASET
import math

def getVocab(src_files, bug_reports, alpha):
    vocab = []
    docs_report = []
    docs_source = []
    vocab_report = []
    for report in bug_reports.values():
        docs_report__ = {}
        data = report.summary
        for word in data:
            if word not in vocab:
                vocab.append(word)
            if word not in vocab_report:
                vocab_report.append(word)

            if word in docs_report__.keys():
                docs_report__[word] += 1
            else:
                docs_report__[word] = 1
        
        docs_report.append(docs_report__)

    for src in src_files.values():
        docs_source_ = {}
        data = src.src_all
        for word in data:
            if word not in vocab:
                vocab.append(word)
            if word in docs_source_.keys():
                docs_source_[word] += 1
            else:
                docs_source_[word] = 1
        docs_source.append(docs_source_)
        
    print(len(vocab))
    print(len(vocab_report))
    tfidf = TFIDFVectorizer()
    report_idf = []
    for doc in docs_report:
        for word in doc:
            doc[word] = tfidf.tfidf(word, doc, docs_report)
        row = []
        for f in vocab:
            if f in doc.keys():
                row.append(doc[word])
            else:
                row.append(0)
        report_idf.append(row)
    print("report_idf")

    source_idf = []
    Maxterm_sf= 0
    Minterm_sf = 100000000
    matrix_g = []
    value_g = []
    for doc in docs_source:
        length_doc = sum(doc.values())
        value_g.append(sum(doc.values()))
        if length_doc > Maxterm_sf:
            Maxterm_sf = length_doc
        if length_doc < Minterm_sf:
            Minterm_sf = length_doc
        for word in doc:
            doc[word] = tfidf.tfidf(word, doc, docs_source)
        row = []
        for f in vocab:
            if f in doc.keys():
                row.append(doc[word])
            else:
                row.append(0)
        source_idf.append(row)
    print("source_idf")
    print([Minterm_sf, Maxterm_sf])
    for value in value_g:
        matrix_g.append(1/( 1+ math.exp(-(value-Minterm_sf)/(Maxterm_sf-Minterm_sf)) ))
    # print(matrix_g)
    report_idf = np.array(report_idf)
    source_idf = np.array(source_idf)
    output = matrix_sim(report_idf, source_idf).tolist()
    cosine_similarity_br = matrix_sim(report_idf, report_idf).tolist()
    with open(DATASET.root / 'name_src_label_1.json', 'rb') as file:
        label_1 = json.load(file)
    sim_simi = []
    for i, b1 in enumerate(cosine_similarity_br):
        sim_x = []
        for s, source in enumerate(src_files.values()):
            sim = 0
            for j, b2 in enumerate(b1):
                if j >= i:
                    break
                if(s in label_1[j]):
                    sim += cosine_similarity_br[i][j]/len(label_1[j])
            sim_x.append(sim)
        sim_simi.append(sim_x)
    out = []
    print(output[1][23], matrix_g[23], sim_simi[1][23])
    print(output[1][60], matrix_g[60], sim_simi[1][60])
    for i, bug in enumerate(output):
        out_x = []
        for j, source in enumerate(bug):
            out_x.append((1 - alpha) * (source * matrix_g[j]) + alpha * sim_simi[i][j])
        out.append(out_x)
    return out

def matrix_sim(R, S):
    d_matrix = np.dot(R, S.T)

    squared_R = np.sqrt(np.sum(R**2, axis=1)).reshape(-1, 1)
    squared_S = np.sqrt(np.sum(S**2, axis=1)).reshape(-1, 1)

    norm = np.dot(squared_R, squared_S.T)

    return np.multiply(d_matrix, 1/norm)

def main():
    
    # Unpickle preprocessed data
    with open(DATASET.root / 'preprocessed_src.pickle', 'rb') as file:
        src_files = pickle.load(file)
    with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
        bug_reports = pickle.load(file)
    simis = getVocab(src_files, bug_reports, 0.2)
    # sm = Similarity(src_files)
    # simis = sm.find_similars(bug_reports)
    
    # Saving similarities in a json file
    with open(DATASET.root / 'rvsm_similarity.json', 'w') as file:
        json.dump(simis, file)
    
    
if __name__ == '__main__':
    main()
