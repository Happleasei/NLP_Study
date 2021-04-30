from pyhanlp import *
# 分词
s = '自然语言处理很有趣'
word_seg = HanLP.segment(s)
# 输出分词结果
print(word_seg)
# ['自然', '语言', '处理', '很', '有趣']