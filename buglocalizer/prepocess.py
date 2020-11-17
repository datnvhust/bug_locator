#coding:utf-8
"""
对bugreport进行预处理，
1.分句
2.分词
3.过滤标点符号
4.过滤停用词
5.提取词干

"""
import nltk
from nltk import sent_tokenize
import numpy as np
import pandas as pd
import re
import inflection
import string
from assets import stop_words, java_keywords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import operator
from functools import reduce
from nltk import regexp_tokenize
from nltk.tokenize import WordPunctTokenizer


# 函数是用来对bug报告分句的，我的想法是：传入一篇文章（即多个句子） 对其进行分句
def getBugReportSent(bugReport_corpus):
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = sen_tokenizer.tokenize(bugReport_corpus)
    return sents

# 函数用来对java源文件进分句，我的想法是：传入Java文件然后在对其使用readline，获得每一行的数据，认为一行
def getJavaSent(javaFile_corpus):
    pass

def clean_sent(string):
    string = re.sub(r"[^a-zA-Z0-9]", " ", string)
    string = re.sub(r",", " , ", string)
    # string = re.sub(r".", " . ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string

def tokenize(sent):
    """Tokenizing bug reports into tokens"""
    sent = clean_sent(sent)
    sent = split_camelcase(sent)
    sent_tokens = nltk.regexp_tokenize(sent,pattern = '\w+|$[\d\.]+|\S+')

    sent_tokens = sent_tokens[2:]
    return sent_tokens

def split_camelcase(sent):
    # 将驼峰和下划线命名的进行分割
    sentence = re.sub('_', ' ', sent)
    p = re.compile(r'([a-z]|\d)([A-Z])')
    sentence = re.sub(p, r'\1 \2', sentence)
    return sentence

def normalize(token_list):
    """Removing punctuation, numbers and also lowercase conversion"""

    # Building a translate table for punctuation and number removal
    # token_list = [re.findall(r'[0-9]+|[a-z]+', s) for s in token_list]
    # token_list = reduce(operator.add, token_list)
    punctnum_table = str.maketrans({c: None for c in string.punctuation + string.digits})
    rawCorpus_punctnum_rem = [token.translate(punctnum_table) for token in token_list]
    token_ed = [token.lower() for token
                          in rawCorpus_punctnum_rem if token]
    return token_ed

def remove_stopwords(token_list):
    """Removing stop words from tokens"""


    token_stopword = [token for token in token_list
                          if token not in stop_words]
    return token_stopword

def remove_java_keywords(token_list):
    """Removing Java language keywords from tokens"""
    token_java_keywords = [token for token in token_list
                          if token not in java_keywords]
    return token_java_keywords

def stem(token_list):
    """Stemming tokens"""
    # Stemmer instance
    stemmer = SnowballStemmer("english")
    token_stem = [stemmer.stem(token) for token in token_list if token]
    return token_stem


def preprocess(sent):

    # 分词
    token = tokenize(sent)

    # 正则化
    token = normalize(token)

    # 移除停用词
    token = remove_stopwords(token)

    # 移除java关键字
    token = remove_java_keywords(token)

    # 词干化
    token = stem(token)

    # 返回不重复的token
    # token_no_loop = sorted(set(token), key=token.index)
    return token



#  如果要使用测试，就运行这里面的
if __name__ == "__main__":
    file_path = 'C:\\Users\\zhangyq\\Desktop\\data\\bugreport\\AspectJ.csv'
    data = pd.read_csv(file_path)
    length = len(data)
    tokens = []
    # token = preprocess('Tester.check(true, Dummy test) Tester.check(true, Dummy test)')
    # print(token)
    for i in range(0, len(data)):
        token = preprocess(data["rawCorpus"][i])
        tokens.append(token)
        print(token)

    #print(tokens)
    # lenth_token = []
    # for i in range(0, len(token)):
    #     lenth_token.append(len(token[i]))
    #
    # print(sum(lenth_token)/1100)

#
# l2 = sorted(set(token),key=token.index)
# print(l2)
# max_length = 0


# dict ={}
# index = 0
# sum_len = 0
# length_list = []
# for i in range(0, length):
#     token = preprocess(data["rawCorpus"][i])
#     dict[data['bug_id'][i]] = len(token)
#
# sorted_dic = sorted(dict.items(),key = lambda x:x[1],reverse = True)
# print(sorted_dic)

# 这一部分是分句的，暂时用不上
# for i in range(0, length):
#     sent = getSent(data["rawCorpus"][i])
#     len_sent = len(sent)
#     dict[data['bug_id']] =
#     sum_len = sum_len +len_sent
#     length_list.append(len_sent)
#     if(max_length < len_sent):
#         max_length = len_sent
#         index = i
# length_list.sort(reverse=True)
# print(index)
# print(length_list[:])
# print("average length is:", str(sum_len/length))
# print("the max length of sent is :", max_length)
# print(getSent(data["rawCorpus"][index]))