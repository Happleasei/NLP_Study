from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# 词形还原工具
lemmatizer = WordNetLemmatizer()
# 输入文本
input_str = "I had a dream"
# 词形还原
output_str = word_tokenize(input_str)
for word in output_str:
   print(lemmatizer.lemmatize(word))
# 输出结果为：
# I have a dream