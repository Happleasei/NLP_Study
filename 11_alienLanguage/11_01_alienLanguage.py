from nltk import word_tokenize

#读取双语文本
pcorpus = dict()
lines_lang1 = open("data_lang1.txt", "r").readlines()
lines_lang2 = open("data_lang2.txt", "r").readlines()

#将双语语料一一对应
for line1, line2 in zip(lines_lang1, lines_lang2):
    sentence1 = tuple(word_tokenize("NULL " + line1.strip("\n").replace('.', '')))
    sentence2 = tuple(word_tokenize("NULL " + line2.strip("\n").replace('.', '')))
    pcorpus[sentence1] = sentence2

#获取词典
lan1_words = []
for key in pcorpus.keys():
    lan1_words.extend(list(key))
lan1_words = list(set(lan1_words))

lan2_words = []
for value in pcorpus.values():
    lan2_words.extend(list(value))
lan2_words = list(set(lan2_words))

#初始化翻译概率
#第一种语言中的每个词都等可能地翻译成第二种语言的每个词
translation_probs = dict()
for word1 in lan1_words:
    value_probs = dict()
    for word2 in lan2_words:
        value_probs[word2] = 1/len(lan2_words)
    translation_probs[word1] = value_probs


#初始化统计计数
count = dict()
total = dict()
for word1 in lan1_words:
    value_count = dict()
    for word2 in lan2_words:
        value_count[word2] = 0
    count[word1] = value_count
for word2 in lan2_words:
    total[word2] = 0

#EM迭代计算
num_epochs = 500
s_total = dict()
for i in range(num_epochs):
    #E步
    for lang_1, lang_2 in pcorpus.items():
        #计算归一化因子
        for word1 in lang_1:
            s_total[word1] = 0
            for word2 in lang_2:
                s_total[word1] += translation_probs[word1][word2]
        #print("s_total:",s_total)
        #计数对齐关系
        for word1 in lang_1:
            for word2 in lang_2:
                count[word1][word2] += (translation_probs[word1][word2] / s_total[word1])
                total[word2] += (translation_probs[word1][word2] / s_total[word1])
        #print("count:",count)
        #print("total:",total)
    #M步
    for word2 in lan2_words:
        for word1 in lan1_words:
            translation_probs[word1][word2] = (count[word1][word2] / total[word2])
