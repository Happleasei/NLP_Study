import numpy as np
import pandas as pd
import csv
import nltk
import re
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
nltk.download("stopwords")
nltk.download("punkt")

def load_data(file_path, title):
    # 读取数据
    df = pd.read_csv(file_path)
    return df[title]

def pre_data(data):
    # 对每篇文本进行分句后，统一放在同一个数组里
    sens = [sent_tokenize(s) for s in data]
    sens = [t for s in sens for t in s]

    # 加载词干提取器
    stemmer = nltk.stem.porter.PorterStemmer()
    # 加载停用词
    stopwords = set(nltk.corpus.stopwords.words("english"))

    # 句子预处理，主要包括字母小写化、去除去用符号、去除停用词
    def pre_sen(sen):
        sen = sen.lower()
        sen = re.sub(r"[^a-zA-Z]", r"", sen)
        pre_sen = [stemmer.stem(w) for w in sen if w not in stopwords]
        assert len(pre_sen) != 0
        return pre_sen

    pre_sens = [pre_sen(sen) for sen in sens]
    # 返回处理后的数据
    assert len(pre_sens) == len(sens)
    return sens, pre_sens

def create_sim_mat(pre_sens, emb_file, emb_size=100):
    # 读取词向量文件
    emb = pd.read_csv(emb_file, sep=' ',
                          header=None, quoting=csv.QUOTE_NONE, index_col=[0])
    dict_word_emb = emb.T.to_dict('series')

    # 将句子转化为句向量
    sens_vec = []
    for s in pre_sens:
        # 句向量的值为句中所有词向量的平均值
        if len(s) != 0:
            v = sum([dict_word_emb[w] for w in s])/(len(s))
        else:
            v = np.zeros((emb_size,))
        sens_vec.append(v)

    # 建立句子与句子之间的相似度矩阵
    n = len(pre_sens)
    sim_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                sim_mat[i][j] = cosine_similarity([sens_vec[i]],
                                                 [sens_vec[j]])[0, 0]
    # 返回相似度矩阵
    return sim_mat

def summarize(sens, sim_mat, topK=10):
    # 根据相似度矩阵建图
    nx_graph = nx.from_numpy_array(sim_mat)
    # 利用pagerank算法给句子（图中节点）评分
    scores = nx.pagerank(nx_graph)
    # 按得分将句子从大到小排列
    ranked_sens = sorted(((scores[i], s) for i, s in enumerate(sens)), reverse=True)
    # 打印topK得分的句子
    for i in range(topK):
        print(ranked_sens[i][1])

def main():
    # 所处理的文本所在的路径及和行头信息
    file_path = "tennis_articles_v4.csv"
    title = "article_text"
    # 词向量文件路径
    emb_file = "glove.6B.100d.txt"
    # 加载数据
    data = load_data(file_path, title)
    # 处理数据
    sens, pre_sens = pre_data(data)
    # 句子相似度矩阵
    sim_mat = create_sim_mat(pre_sens, emb_file)
    # 获取摘要
    summarize(sens, sim_mat)
    