#coding=GBK
'''
Created on 2019年3月13日

@author: Admin
'''
import jieba 
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.cluster import KMeans
if __name__ == '__main__':
    labels=[] 
    corpus=[]
    stopWords=open('stop.txt','r')
    texts=['\n',' '] 
    for word in stopWords:
        word=word.strip()
        texts.append(word)
    doc=open('suggest.txt','r')
    for line in doc.readlines():
        line=line.strip()
        if line!="":
            labels.append(line)
            data=jieba.cut(line) 
            data_adj=''
            delete_word=[]
            for item in data:
                if item not in texts: 
                    data_adj+=item+' '
                else:
                    delete_word.append(item) 
            corpus.append(data_adj) 
    vectorizer=CountVectorizer()
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    weight=tfidf.toarray()
    mykms=KMeans(n_clusters=20)
    y=mykms.fit_predict(weight)
    for i in range(0,7):
        label_i=[]
        for j in range(0,len(y)):
            if y[j]==i:
                label_i.append(labels[j])
        print('lable_'+str(i)+':'+str(label_i))
        fn = 'lable_'+str(i)
        file=open('result_'+fn+'.txt','w')
        file.write(str(label_i))
        file.close


