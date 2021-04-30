from stanfordcorenlp import StanfordCoreNLP
# 加载分词工具
nlp_model = StanfordCoreNLP(r'stanford-corenlp-full-2018-02-27', lang='zh')
# 分词
s = '自然语言处理很有趣'
word_seg = nlp_model.word_tokenize(s)
# 输出分词结果
print(word_seg)
# ['自然', '语言', '处理', '很', '有趣']