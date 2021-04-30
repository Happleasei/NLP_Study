from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba

raw_texts = [
    '你站在桥上看风景',
    '看风景的人在楼上看你',
    '明月装饰了你的窗子',
    '你装饰了别人的梦',
]
texts = [" ".join(jieba.lcut(text, cut_all=True)) for text in raw_texts]

#tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
#bow_vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_vec = TfidfVectorizer()
bow_vec = CountVectorizer()
bow_matrix = bow_vec.fit_transform(texts)
tfidf_matrix = tfidf_vec.fit_transform(texts)
print(bow_vec.vocabulary_)
print(tfidf_vec.vocabulary_)
print(tfidf_matrix)
print(bow_matrix)
