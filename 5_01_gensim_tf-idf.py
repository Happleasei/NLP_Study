from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import jieba

raw_texts = [
    '你站在桥上看风景',
    '看风景的人在楼上看你',
    '明月装饰了你的窗子',
    '你装饰了别人的梦',
]
texts = [[word for word in jieba.cut(text, cut_all=True)] for text in raw_texts]
print(texts)
dictionary = Dictionary(texts)
print(dictionary.token2id)
bow_texts = [dictionary.doc2bow(text) for text in texts]
print(bow_texts)

tfidf = TfidfModel(bow_texts)
tfidf_vec = [tfidf[text] for text in bow_texts]
print(tfidf_vec)
