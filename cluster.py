from nltk.corpus import stopwords
import json
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

def readIn(path):
    #读入数据并得到key-value的文本以及所属类的形式
    f = open(path)
    line = f.readline()
    all_text = {}
    class_number = []
    while line:
        # print(line)
        my_dict = json.loads(line)
        key = my_dict['text']
        value = my_dict['cluster']
        all_text[key] = value
        class_number.append(value)
        line = f.readline()
    return all_text,len(set(class_number))

def getTFIDF(path):
    #处理得到tf-idf矩阵
    all_text, number_of_class = readIn(path)
    class_number = []
    text_list = []
    for k, v in all_text.items():
        text_list.append(k)
        class_number.append(v)
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(text_list)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    # 将词频矩阵X统计成TF-IDF值
    tf_idf_matrix = transformer.fit_transform(X)
    return tf_idf_matrix,number_of_class,class_number

def getNMI(method,res,class_number):
    nmi = metrics.normalized_mutual_info_score(res, class_number)
    print("Method: ",method)
    print("NMI: ",nmi)

def kmeansClustering(tf_idf_matrix,num_clusters,class_number):
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++')
    # 返回各自文本的所被分配到的类索引
    res = km_cluster.fit_predict(tf_idf_matrix)
    getNMI("K-Means",res,class_number)
    print("The number of clusters: ",num_clusters)

def affinityPropagationClustering(tf_idf_matrix,class_number):
    af = AffinityPropagation(affinity='euclidean', preference=-32).fit(tf_idf_matrix)  # preference越小聚类数目越少
    res = af.fit_predict(tf_idf_matrix)
    cluster_centers_indices = af.cluster_centers_indices_
    num_clusters = len(cluster_centers_indices)
    getNMI("Affinity propagation",res,class_number)
    print("The number of clusters: ", num_clusters)

def meanShiftClustering(tf_idf_matrix,class_number):
    ms = MeanShift(bandwidth=0.67, bin_seeding=True)  # 3.2
    ms.fit_predict(tf_idf_matrix.toarray())
    res = ms.labels_
    cluster_centers = ms.cluster_centers_
    num_clusters = len(cluster_centers)
    getNMI("MeanShift",res,class_number)
    print("The number of clusters: ", num_clusters)

def spectralClustering(tf_idf_matrix,num_clusters,class_number):
    sc = SpectralClustering(n_clusters=num_clusters)
    res = sc.fit_predict(tf_idf_matrix)
    getNMI("Spectral clustering",res,class_number)
    print("The number of clusters: ",num_clusters)

def agglomerativeClustering(tf_idf_matrix,num_clusters,class_number):
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    res = ac.fit_predict(tf_idf_matrix.todense())
    getNMI("Agglomerative ward clustering",res,class_number)
    print("The number of clusters: ",num_clusters)

def agglomerativeComplete(tf_idf_matrix,num_clusters,class_number):
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='complete')
    res = ac.fit_predict(tf_idf_matrix.todense())
    getNMI("Agglomerative complete clustering",res,class_number)
    print("The number of clusters: ",num_clusters)

def agglomerativeAverage(tf_idf_matrix,num_clusters,class_number):
    ac = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='average')
    res = ac.fit_predict(tf_idf_matrix.todense())
    getNMI("Agglomerative average clustering",res,class_number)
    print("The number of clusters: ",num_clusters)

def DBSCANClustering(tf_idf_matrix,class_number):
    db = DBSCAN(eps=1.1, min_samples=3)
    res = db.fit_predict(tf_idf_matrix.toarray())
    getNMI("DBSCAN",res,class_number)
    num_clusters =len(set(db.labels_))
    print("The number of clusters: ", num_clusters)

def gaussianMixtureClustering(tf_idf_matrix,num_clusters,class_number):
    gm = GaussianMixture(n_components=num_clusters, covariance_type='diag', max_iter=200, random_state=0)
    gm.fit(tf_idf_matrix.toarray())
    res = gm.predict(tf_idf_matrix.toarray())
    getNMI("Gaussian mixture clustering",res,class_number)
    print("The number of clusters: ",num_clusters)

path = "C:\data\Tweets.txt"
tf_idf_matrix,num_clusters,class_number = getTFIDF(path)
#kmeansClustering(tf_idf_matrix,num_clusters,class_number)
affinityPropagationClustering(tf_idf_matrix,class_number)
#meanShiftClustering(tf_idf_matrix,class_number)
#spectralClustering(tf_idf_matrix,num_clusters,class_number)
#agglomerativeClustering(tf_idf_matrix, num_clusters, class_number)
#agglomerativeComplete(tf_idf_matrix, num_clusters, class_number)
#agglomerativeAverage(tf_idf_matrix,num_clusters,class_number)
#DBSCANClustering(tf_idf_matrix,class_number)
#gaussianMixtureClustering(tf_idf_matrix,num_clusters,class_number)