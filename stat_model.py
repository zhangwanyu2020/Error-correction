import re
import collections

# 获取词表、计数
with open('/Users/zhangwanyu/Desktop/NLP课程/NLP_week_17/pt20200913/data.txt') as f:
    data = f.readlines()
    model = collections.defaultdict(int)
    vocab = set()
    for line in data:
        line = re.findall("[a-z]+",line.lower())
        for word in line:
            model[word] += 1
            vocab.add(word)
print(len(model)) # 29154
print(len(vocab)) # 29154

alphabet = "abcdefghijklmnopqrstuvwxyz"
def filter(words):
    new_words = set()
    for word in words:
        if word in vocab:
            new_words.add(word)
    return new_words

# 增删改1个字符
def edist_1(word):
    n = len(word)
    # 删除 n种情况
    word1 = [word[0:i] + word[i+1:] for i in range(n)]
    # 增加 n*len(alphabet)种情况
    word2 = [word[0:i] + c + word[i+1:] for i in range(n) for c in alphabet]
    # 相邻交换 n-1种情况
    word3 = [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1)]
    # 替换
    word4 = [word[0:i] + c + word[i+1:] for i in range(n) for c in alphabet]
    words = set(word1+word2+word3+word4)
    return filter(words)

def edist_2(word):
    words = set()
    for w in edist_1(word):
        word_2 = edist_1(word)
        words.add(word_2)
    return words

def correct(word):
    if word not in vocab:
        candidates = edist_1(word) or edist_2(word)
        print(candidates)
        return max(candidates,key=lambda w:model[w])
    else:
        return word
res = correct('mske')
print('正确答案是：',res)



