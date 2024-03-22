from nltk import word_tokenize, pos_tag, ne_chunk
# 输入文本
input_str = "Bill works for Apple so he went to Boston for a conference."
# 命名实体识别
print(ne_chunk(pos_tag(word_tokenize(input_str))))
# 输出结果为：
# [(‘Dive’, ‘JJ’), (‘into’, ‘IN’), (‘NLTK’, ‘NNP’), (‘:’, ‘:’), (‘Part-of-speech’, ‘JJ’), (‘tagging’, ‘NN’),
# (‘and’, ‘CC’), (‘POS’, ‘NNP’), (‘Tagger’, ‘NNP’)]