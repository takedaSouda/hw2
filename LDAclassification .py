#First you should add input dataset. My dataset is open access via https://www.kaggle.com/datasets/xianyumaomao/miaohomewook?utm_medium=social&utm_campaign=kaggle-dataset-share&utm_source=twitter

import os
import re
import jieba
from collections import Counter

novel_labels = {
    '白马啸西风': 1,
    '碧血剑': 2,
    '飞狐外传': 3,
    '连城诀': 4,
    '鹿鼎记': 5,
    '三十三剑客图': 6,
    '射雕英雄传': 7,
    '神雕侠侣': 8,
    '书剑恩仇录': 9,
    '天龙八部': 10,
    '侠客行': 11,
    '笑傲江湖': 12,
    '雪山飞狐': 13,
    '倚天屠龙记': 14,
    '鸳鸯刀': 15,
    '越女剑': 16
}

def get_corpus_jieba(rootDir, max_tokens_per_segment=1000, total_segments=1000):
    corpus = []
    labels = []
    i=0
    
    with open("/kaggle/input/miaohomewook/cn_stopwords.txt", "r",encoding="utf-8") as file:
        stopword_list = [line.strip() for line in file]

    listdir = os.listdir(rootDir)
    num_files = len([f for f in listdir if f.endswith('.txt')])
    segments_per_file = max(50, total_segments // num_files + num_files)  # Ensure at least one segment per file

    for filename in listdir:
        corpu = []
        if filename.endswith('.txt'):
            path = os.path.join(rootDir, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding='gbk', errors='ignore') as file:
                    filecontext = file.read()
                    filecontext = ' '.join(word for word in filecontext.split() if word not in stopword_list)
                    filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    tokens = list(jieba.cut(filecontext))
                    segment = []
                    count = 0
                    for token in tokens:
                        segment.append(token)
                        count += 1
                        if count >= max_tokens_per_segment:
                            corpus.append(segment)
                            corpu.append(segment)
                            labels.append(novel_labels[filename.replace('.txt', '')])  # Use the file name as the label
                            segment = []
                            count = 0  # Reset the counter for the next segment
                            if len(corpu) >= segments_per_file:
                                corpu=[]
                                break  # Stop after adding required number of segments
    

    from collections import Counter
    import matplotlib.pyplot as plt
    label_counts = Counter(labels)
    print(label_counts)


    return corpus, labels

def get_corpus_word(rootDir, max_tokens_per_segment=1000, total_segments=1000):
    corpus = []
    labels = []
    i=0
    
    with open("/kaggle/input/miaohomewook/cn_stopwords.txt", "r",encoding="utf-8") as file:
        stopword_list = [line.strip() for line in file]

    listdir = os.listdir(rootDir)
    num_files = len([f for f in listdir if f.endswith('.txt')])
    segments_per_file = max(50, total_segments // num_files + num_files)  # Ensure at least one segment per file

    for filename in listdir:
        corpu = []
        if filename.endswith('.txt'):
            path = os.path.join(rootDir, filename)
            if os.path.isfile(path):
                with open(path, "r", encoding='gbk', errors='ignore') as file:
                    filecontext = file.read()
                    filecontext = ' '.join(word for word in filecontext.split() if word not in stopword_list)
                    filecontext = filecontext.replace("\n", '')
                    filecontext = filecontext.replace(" ", '')
                    segment = []
                    count = 0
                    for char in filecontext:
                        segment.append(char)
                        count += 1
                        if count >= max_tokens_per_segment:
                            corpus.append(segment)
                            corpu.append(segment)
                            labels.append(novel_labels[filename.replace('.txt', '')])  # Use the file name as the label
                            segment = []
                            count = 0  # Reset the counter for the next segment
                            if len(corpu) >= segments_per_file:
                                corpu=[]
                                break  # Stop after adding required number of segments
    

    from collections import Counter
    import matplotlib.pyplot as plt
    label_counts = Counter(labels)
    print(label_counts)


    return corpus, labels


# rootDir = 'AugmentTest/textTest/NLP_homewrok-main/data'
# corpus, labels = get_corpus_jieba(rootDir,max_tokens_per_segment=1000,total_segments=1000)
# features = get_LDA_features(corpus, num_topics=128, passes=10)

from gensim import corpora, models
import numpy as np
# 假设 corpus 是已经分好词的文档列表
# 创建字典
def get_LDA_features(corpus,num_topics = 10,passes=10):
    dictionary = corpora.Dictionary(corpus)
# 通过字典将文档转换为词袋模型
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
# 指定LDA模型的主题数 T
    num_topics = num_topics # 根据需求设置这个值
# 训练LDA模型
    lda = models.LdaModel(corpus_bow, num_topics=num_topics, id2word=dictionary, passes=passes)
# 为每个段落计算主题分布
    topic_distributions = [lda.get_document_topics(bow, minimum_probability=0.0) for bow in corpus_bow]
# 将主题分布转换为固定长度的向量
    topic_features = np.zeros((len(corpus), num_topics))
    for i, doc_topics in enumerate(topic_distributions):
        for topic in doc_topics:
            topic_id, prob = topic  # 解包元组
            topic_features[i, topic_id] = prob
    return topic_features



# labels = np.array(labels)
# svm_classifier = SVC(kernel='linear', probability=True)  # 使用线性核
# scores = cross_val_score(svm_classifier, features, labels, cv=10)




import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# from kaggle.input.miaohomewook.pre_data import get_corpus_jieba,get_corpus_word
# from kaggle.input.miaohomewook.LDA import get_LDA_features


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rootDir = '/kaggle/input/miaohomewook/data'


max_tokens_values = [20,100, 500, 1000, 3000]
mean_scores = []
std_scores=[]

for max_tokens_per_segment in max_tokens_values:
    corpus, labels = get_corpus_jieba(rootDir, max_tokens_per_segment=max_tokens_per_segment, total_segments=1000)
    features = get_LDA_features(corpus, num_topics=64, passes=10)
    labels = np.array(labels)

    svm_classifier = SVC(kernel='linear', probability=True)
    lr_classifier=LogisticRegression()
    dc_classifier=DecisionTreeClassifier()
    scores = cross_val_score(dc_classifier, features, labels, cv=10)
    print(scores.mean())
    mean_scores.append(scores.mean())
    std_scores.append(scores.std())

# 绘制曲线图
plt.plot(max_tokens_values, mean_scores, marker='o')
plt.xlabel('max_tokens_per_segment')
plt.ylabel('Average Accuracy')
plt.title('Performance vs. max_tokens_per_segment')
plt.grid(True)
plt.show()
