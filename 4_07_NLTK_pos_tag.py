import nltk
# 输入文本
input_str = "Dive into NLTK: Part-of-speech tagging and POS Tagger"
# 词性标注
tokens = nltk.word_tokenize(input_str)
output = nltk.pos_tag(tokens)
print(output)
# 输出结果为：
# [(‘Dive’, ‘JJ’), (‘into’, ‘IN’), (‘NLTK’, ‘NNP’), (‘:’, ‘:’), (‘Part-of-speech’, ‘JJ’), (‘tagging’, ‘NN’), (‘and’, ‘CC’), (‘POS’, ‘NNP’), (‘Tagger’, ‘NNP’)]