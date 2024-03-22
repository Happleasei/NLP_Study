import re,collections

# 提取语料库中的所有单词并且转化为小写
def words(text):
    return re.findall("[a-z]+", text.lower())

# 若单词不在语料库中，默认词频为1，避免先验概率为0的情况
def train(features):
    model = collections.defaultdict(lambda:1)#若key为空，默认值为1
    for f in features:
        model[f]+=1#统计词频
    return model

words_N = train(words(open("bayes_train_text.txt").read()))

#英文字母
alphabet="abcdefghijklmnopqrstuvwxyz"

# 编辑距离为1的所有单词
def edits1(word):
    n = len(word)
    # 删除某一字母而得的词
    s1 = [word[0:i]+word[i+1:] for i in range(n)]
    # 相邻字母调换位置
    s2 = [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)]
    # 替换
    s3 = [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet]
    # 插入
    s4 = [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet]
    edits1_words = set(s1+s2+s3+s4)
    return edits1_words

# 编辑距离为2的所有单词
def edits2(word):
    edits2_words = set(e2 for e1 in edits1(word) for e2 in edits1(e1))
    return edits2_words

# 过滤非词典中的单词
def known(words):
    return set(w for w in words if w in words_N)

def correct(word):
    if word not in words_N:
        candidates = known(edits1(word)) | known(edits2(word))
        return max(candidates, key=lambda w:words_N[w])
    else:
        return None

print(correct("het"))
print(correct("annd"))
