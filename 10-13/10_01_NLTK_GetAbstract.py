import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
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

def topNwords(pre_sens, n=50):
    # 输出文本数据中前N个频次高的词
    words = [w for s in pre_sens for w in s]
    word_fre = nltk.FreqDist(words)
    topN_words = [w[0] for w in sorted(word_fre.items(), key=lambda d:d[1], reverse=True)][:n]
    return topN_words

def sen_score(sen, topN_words, cluster_threshold):
    # 计算每个句子的得分
    # 标记每个句子中出现topN中单词的位置
    word_idx = []
    for w in topN_words:
        try:
            word_idx.append(sen.index(w))
        except ValueError:
            pass
    word_idx.sort()

    # 如果句子中不存在topN中的单词，得分为0
    if len(word_idx) == 0:
        return 0

    # 根据句子中topN单词出现的位置将句子转化为簇的集合形式
    clusters = []
    cluster = [word_idx[0]]
    i = 1
    while i < len(word_idx):
        # 当topN单词间的距离小于一定的阈值时，这些单词及其间非topN单词共同形成簇
        if word_idx[i] - word_idx[i-1] < cluster_threshold:
            cluster.append(word_idx[i])
        else:
            clusters.append(cluster)
            cluster = [word_idx[i]]
        i += 1

    clusters.append(cluster)

    # 计算每个句子中所有簇的得分值，最高值即为句子得分
    max_score = 0
    for c in clusters:
        # 每个簇中topN单词的个数
        words_important = len(c)
        # 每个簇所有单词的个数
        words_total = c[-1]-c[0]+1
        # 得分计算公式
        score = words_important**2/words_total
        if score > max_score:
            max_score = score
    # 返回句子得分
    return max_score

def main():
    # 所处理的文本所在的路径及和行头信息
    file_path = "tennis_articles_v4.csv"
    title = "article_text"
    # 设置topN词语间的距离阈值
    cluster_threshold = 5
    # 设置摘要句子个数
    topK = 10
    # 加载数据
    data = load_data(file_path, title)
    # 处理数据
    sens, pre_sens = pre_data(data)
    # 计算topN
    topN_words = topNwords(pre_sens)

    # 计算每个句子得分
    scores = []
    for i, pre_sen in enumerate(pre_sens):
       score = sen_score(pre_sen, topN_words, cluster_threshold)
       sen = sens[i]
       scores.append((score, sen))

    sorted(scores, reverse=True)
    # 获取摘要
    for i in range(topK):
        print(scores[i][1])
