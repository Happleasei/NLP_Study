from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# 词干提取工具
stemmer = PorterStemmer()
# 输入文本
input_str = "There are several types of stemming algorithms"
# 词干提取
output_str = word_tokenize(input_str)
for word in output_str:
   print(stemmer.stem(word))
# 输出结果为：
# There are sever type of stem algorithm