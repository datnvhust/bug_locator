import pickle
import json

from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from datasets import DATASET
import math


class Similarity:
    
    __slots__ = ['src_files', 'src_strings']
    
    def __init__(self, src_files):
        self.src_files = src_files
        self.src_strings = [' '.join(src.file_name['stemmed'] + src.class_names['stemmed']
                                     + src.method_names['stemmed'] + src.pos_tagged_comments['stemmed']
                                     + src.attributes['stemmed'])
                            for src in self.src_files.values()]
    
    def calculate_similarity(self, src_tfidf, reports_tfidf):
        """Calculatnig cosine similarity between source files and bug reports"""
        # Normalizing the length of source files
        src_lenghts = np.array([float(len(src_str.split()))
                                for src_str in self.src_strings]).reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_src_len = min_max_scaler.fit_transform(src_lenghts)
         
        # Applying logistic length function
        src_len_score = 1 / (1 + np.exp(-12 * normalized_src_len))
        print(len(src_len_score))
        simis = []
        for report in reports_tfidf:
            s = cosine_similarity(src_tfidf, report)
            
            # revised VSM score calculation
            rvsm_score = s * src_len_score
            
            normalized_score = np.concatenate(
                min_max_scaler.fit_transform(rvsm_score)
            )
            
            simis.append(normalized_score.tolist())
            
        return simis
    
    def calculate_similarity_br(self, reports_tfidf, bug_reports):
        """Calculatnig cosine similarity between source files and bug reports"""
        # Normalizing the length of source files
        reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed'])
                           for report in bug_reports.values()]
        src_lenghts = np.array([float(len(src_str.split()))
                                for src_str in reports_strings]).reshape(-1, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        normalized_src_len = min_max_scaler.fit_transform(src_lenghts)
         
        # Applying logistic length function
        src_len_score = 1 / (1 + np.exp(-12 * normalized_src_len))
        print(len(src_len_score))
        simis = []
        for report in reports_tfidf:
            s = cosine_similarity(reports_tfidf, report)
            
            # revised VSM score calculation
            rvsm_score = s * src_len_score
            
            normalized_score = np.concatenate(
                min_max_scaler.fit_transform(rvsm_score)
            )
            
            simis.append(normalized_score.tolist())
            
        return simis
    
    def find_similars(self, bug_reports, src_files, alpha):
        """Calculating tf-idf vectors for source and report sets
        to find similar source files for each bug report.
        """
        
        reports_strings = [' '.join(report.summary['stemmed'] + report.description['stemmed'])
                           for report in bug_reports.values()]
        
        tfidf = TfidfVectorizer(sublinear_tf=True, smooth_idf=False)
        src_tfidf = tfidf.fit_transform(self.src_strings)
        
        reports_tfidf = tfidf.transform(reports_strings)
        report_idf = reports_tfidf.toarray().tolist()
        source_idf = src_tfidf.toarray().tolist()

        docs_source = []
        for src in src_files.values():
            docs_source_ = {}
            data = src.src_all
            for word in data:
                if word in docs_source_.keys():
                    docs_source_[word] += 1
                else:
                    docs_source_[word] = 1
            docs_source.append(docs_source_)
            
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
        print([Minterm_sf, Maxterm_sf])
        for value in value_g:
            matrix_g.append(1/( 1+ math.exp(-(value-Minterm_sf)/(Maxterm_sf-Minterm_sf)) ))
        report_idf = np.array(report_idf)
        source_idf = np.array(source_idf)
        # output = matrix_sim(report_idf, source_idf).tolist()
        output = self.calculate_similarity(src_tfidf, reports_tfidf)
        # print(output[1])
        # print(output1[1])
        # cosine_similarity_br = matrix_sim(report_idf, report_idf).tolist()
        cosine_similarity_br = self.calculate_similarity_br(reports_tfidf, bug_reports)
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
        simis = self.calculate_similarity(src_tfidf, reports_tfidf)
        return output

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
    
    sm = Similarity(src_files)
    simis = sm.find_similars(bug_reports, src_files, 0.2)
    
    # Saving similarities in a json file
    with open(DATASET.root / 'vsm_similarity.json', 'w') as file:
        json.dump(simis, file)
    
    
if __name__ == '__main__':
    main()
