import thulac
# 加载分词工具
thulac_model = thulac.thulac()
# 分词
s = '自然语言处理很有趣'
word_seg = thulac_model.cut(s)
# 输出分词结果
print(word_seg)
# [['自然', 'n'], ['语言', 'n'], ['处理', 'v'], ['很', 'd' ], ['有趣', 'a']]