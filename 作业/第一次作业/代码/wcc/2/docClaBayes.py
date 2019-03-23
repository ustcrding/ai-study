#coding:UTF-8
'''
Created on 2019年3月13日
@author: Admin
'''
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import pandas
if __name__ == '__main__':
    labels=[]
    corpus=[]
    stopWords=open('data/stop.txt','r')
    texts=['\n',' ']
    for word in stopWords:
        word=word.strip()
        texts.append(word)
    doc=open('data/data.txt','r')
    for line in doc.readlines():
        x,y=line.strip().split(',',1)
        labels.append(x)
        data=jieba.cut(y)
        data_adj=''
        for item in data:
            if item not in texts:
                data_adj+=item+' '
        corpus.append(data_adj)
    trainDF = pandas.DataFrame()
    trainDF['text'] = corpus
    trainDF['label'] = labels
    train_x,test_x,train_y,test_y=model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.2,random_state=42)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',stop_words=stopWords)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf=tfidf_vect.transform(train_x)
    xtest_tfidf=tfidf_vect.transform(test_x)
    classifier=naive_bayes.MultinomialNB()
    classifier.fit(xtrain_tfidf, train_y)
    predictions = classifier.predict(xtest_tfidf)
    accuracy=metrics.accuracy_score(predictions, test_y)
    print ("TF-IDF-朴素贝叶斯，预测准确率:%.1f "%accuracy);









