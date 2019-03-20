# -*- coding: utf-8 -*-
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import operator
import time
from numpy import *
from wordcloud import WordCloud


filename = "reqlog3.txt"
corpus = []  # 空语料库
text_list = []


# 切词
def jieba_tokenize(text):
    return jieba.lcut(text)

def init_jieba(userdict_filepath = "userdict.txt"):
    jieba.load_userdict(userdict_filepath)

# 对句子进行分词
def segment(text, stop_words):
    seg_list = jieba.cut(text, cut_all=False)
    #seg_list = jieba.cut(text)
    seg_list_without_stopwords = ''
    for word in seg_list:
        if word not in stop_words:
            if word != '\t':
                # seg_list_without_stopwords.append(word)
                seg_list_without_stopwords += word + ' '
    return seg_list_without_stopwords

def remove_stopword(text, stop_words):
    data = jieba.cut(text)  # 文本分词
    data_adj = ''
    delete_word = []
    for item in data:
        if item not in stop_words:  # 停用词过滤
            data_adj += item + ' '
        else:
            delete_word.append(item)

    return data_adj;

def load_stopwords(stop_word_file_path):
    '''停用词的过滤'''
    typetxt = open(stop_word_file_path, "r", encoding="utf-8")
    texts = ['\u3000', '\n', ' ']  # 爬取的文本中未处理的特殊字符
    '''停用词库的建立'''
    for word in typetxt:
        word = word.strip()
        texts.append(word)
    return texts

def load_data(filename, stop_words):
    f = open(filename, encoding='utf-8')
    line = f.readline()  # 调用文件的 readline()方法

    text_list = []
    corpus = []
    if line is not None:
        line.strip()
        line = line.replace("\n",'')
        text_list.append(line)
        corpus.append(segment(line, stop_words))
    while line:
        line = f.readline()
        line.strip()
        line = line.replace("\n", '')
        text_list.append(line)
        corpus.append(segment(line, stop_words))
    f.close()

    return text_list, corpus

def main():
    stop_word_file_path = 'stop_words.txt'
    init_jieba('userdict.txt')
    stop_words = load_stopwords(stop_word_file_path)
    text_list, corpus = load_data(filename, stop_words)

    '''
    tokenizer: 指定分词函数
    lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
    所以最好是False
    '''
    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer()
    # 该类会统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    tfidf_matrix = tfidf.toarray()
    # 获取词袋模型中的所有词
    word = vectorizer.get_feature_names()

    '''
    n_clusters: 指定K的值
    max_iter: 对于单次初始值计算的最大迭代次数
    n_init: 重新选择初始值的次数
    init: 制定初始值选择的算法
    n_jobs: 进程个数，为-1的时候是指默认跑满CPU
    注意，这个对于单个初始值的计算始终只会使用单进程计算，
    并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40, 
    服务器上面有20个CPU可以开40个进程，最终只会开10个进程
    '''
    num_clusters = 20
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=1, init='k-means++', n_jobs=1)

    # 返回各自文本的所被分配到的类索引
    print(tfidf_matrix)
    result = km_cluster.fit_predict(tfidf_matrix)
    print("Predicting result: ", result)
    for i in range(0, num_clusters):
        label_i = []
        for j in range(0, len(result)):
            if result[j] == i:
                label_i.append(text_list[j])
        print('label_' + str(i) + ':' + str(label_i))

    # plotFeature(tfidf_matrix, result, clustAssing)

    # 可视化操作
    x = [n[0] for n in tfidf_matrix]
    y = [n[1] for n in tfidf_matrix]
    plt.scatter(x, y, c=result, marker='x')
    plt.title("Kmeans-Basketball Data")
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(["Rank"])
    # plt.show()

    makeWordCloud(text_list)

def plotFeature(dataSet, centroids, clusterAssment):
    m = shape(centroids)[0]
    fig = plt.figure()
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    scatterColors = ['blue', 'green', 'yellow', 'purple', 'orange', 'black', 'brown']
    ax = fig.add_subplot(111)
    for i in range(m):
        ptsInCurCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        colorSytle = scatterColors[i % len(scatterColors)]
        ax.scatter(ptsInCurCluster[:, 0].flatten().A[0], ptsInCurCluster[:, 1].flatten().A[0], marker=markerStyle, c=colorSytle, s=90)
    ax.scatter(centroids[:, 0].flatten().A[0], centroids[:, 1].flatten().A[0], marker='+', c='red', s=300)

def makeWordCloud(text_list):
    all_text = ""
    for str in text_list:
        all_text += str

    # cut_text = " ".join(jieba.cut(all_text))
    # wordcloud = WordCloud().generate(cut_text)
    result = jieba.analyse.textrank(all_text, topK=300, withWeight=True)
    keywords = dict()
    for i in result:
        keywords[i[0]] = i[1]
    wordcloud = WordCloud(stopwords='stop_words.txt').generate_from_frequencies(keywords)
    import matplotlib.pyplot as plt
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print('finish all in %s' % str(end - start))
    plt.show()
