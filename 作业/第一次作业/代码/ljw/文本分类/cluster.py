import jieba
from sklearn.cluster import KMeans
import numpy as np
import os


def read_from_file(file_name):
    with open(file_name, "r", encoding="utf-8") as fp:
        words = fp.read()
    return words


def stop_words(stop_word_file):
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        new_words.append(r)
    return set(new_words)


def del_stop_words(words, stop_words_set):
    #   words是已经切词但是没有去除停用词的文档。
    #   返回的会是去除停用词后的文档
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stop_words_set:
            new_words.append(r)
    return new_words


def get_all_vector(file_path, stop_words_set):
    posts = open(file_path, encoding="utf-8").read().split("\n")
    docs = []
    word_set = set()
    for post in posts:
        doc = del_stop_words(post, stop_words_set)
        docs.append(doc)
        word_set |= set(doc)
        # print len(doc),len(word_set)

    word_set = list(word_set)
    docs_vsm = []
    # for word in word_set[:30]:
    # print word.encode("utf-8"),

    for doc in docs:
        temp_vector = []
        for word in word_set:
            temp_vector.append(doc.count(word) * 1.0)
        # print temp_vector[-30:-1]
        docs_vsm.append(temp_vector)

    docs_matrix = np.array(docs_vsm)
    column_sum = [float(len(np.nonzero(docs_matrix[:, i])[0])) for i in range(docs_matrix.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    idf = np.log(column_sum)
    idf = np.diag(idf)
    # 请仔细想想，根绝IDF的定义，计算词的IDF并不依赖于某个文档，所以我们提前计算好。
    # 注意一下计算都是矩阵运算，不是单个变量的运算。
    for doc_v in docs_matrix:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())

    tfidf = np.dot(docs_matrix, idf)
    return posts, tfidf


if __name__ == "__main__":
    stop_words = stop_words("./stopwords.txt")
    names, tfidf_mat = get_all_vector("./意见反馈.txt", stop_words)
    km = KMeans(n_clusters=10)
    km.fit(tfidf_mat)
    clusters = km.labels_.tolist()
    str_clusters = {}
    for i in range(len(clusters)):
        if str_clusters.get(clusters[i]) is None:
            str_clusters.setdefault(clusters[i], [])
        str_clusters.get(clusters[i]).append(names[i])
