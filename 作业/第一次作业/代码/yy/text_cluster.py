import jieba 
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

def jieba_tokenize(text):
    return jieba.lcut(text)


def main():
    textFile = r'意见反馈.txt'
    rstFile='结果.txt'
    tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, lowercase=False)
    with open(textFile, 'r') as f:
        text_list=f.readlines()
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)
    num_clusters = 200
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=3000, n_init=1, \
                    init='k-means++',n_jobs=1)
    result = km_cluster.fit_predict(tfidf_matrix)
    rstdf=pd.DataFrame({'clu_index':result,
                     'text_list':text_list})
    rstdf.sort_values('clu_index', inplace=True)
    with open(rstFile, 'w') as fw:
        fw.writelines(rstdf['text_list'])


if __name__ == '__main__':
    main()
exit

