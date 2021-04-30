import jieba
s = '自然语言处理很有趣'
# 全模式分词
wordseg_all = jieba.lcut(s, cut_all=True)
# 输出分词结果
print(wordseg_all)
# ['自然', '自然语言', '语言', '处理', '很', '有趣']

# 精确模式分词
wordseg = jieba.lcut(s, cut_all=False)
# 输出分词结果
print(wordseg)
# ['自然语言', '处理', '很', '有趣']

# 搜索引擎模式分词
wordseg_search = jieba.lcut_for_search(s)
print(wordseg_search)
# ['自然', '语言', '自然语言', '处理', '很', '有趣']