import heapq
import cmath
import numpy as np
count_right = 0
count_wrong = 0
K = 20

def classify( train_vec, dic, test_vec, files):
    print('classify....')
    train_space = get_all_key_value(dic, train_vec)
    test_space = get_all_key_value(dic, test_vec)
    i =0
    for each_test_space in test_space:
        original = files[i]
        count_right, count_wrong =  one_test_all_train( each_test_space, train_space,original, files )
        i+=1
    return count_right,count_wrong
#某一测试组比对所有训练组进行分类
def one_test_all_train( each_test_space, train_space,original, files):
    print('one_test_all_train....')
    global count_right, count_wrong
    group_all_key = []
    for doc_space in each_test_space:
        doc_all_sim = doc_all_sim_cal(doc_space,train_space,files)
        # g找前k个最大相似度判断其属于哪一类
        klargest_sim = heapq.nlargest(K, doc_all_sim.items(), key=lambda s: s[0])
        # 计算这个文档属于哪一类
        largest = count_max(klargest_sim)
        classfied = list(largest[0])[0]
        print('This doc is classfied to ', classfied, ' ,原属于' + original)
        if classfied == original:
            count_right = count_right + 1
        else:
            count_wrong = count_wrong + 1

    return count_right,count_wrong

def doc_all_sim_cal(doc_space,train_space,files):
    count_group = 0
    all_sim = {}
    for group in train_space:
        for train_doc in group:
            fenzi = 0
            fenmu1 = 0
            for key,value in train_doc.items():
                try:
                    doc_value = doc_space[key]
                except KeyError:
                    doc_value = 0
                else:
                    doc_value = doc_value
                fenmu1 += value*value;
                fenzi += value * doc_value
                fenmu2 = 0
                for dk,dv in doc_space.items():
                    fenmu2 += dv*dv
                fenmu = (cmath.sqrt(fenmu1)*cmath.sqrt((fenmu2)))
                if(np.real((fenmu))>0):
                    doc_doc_sim = fenzi/fenmu*1000
                    all_sim[str(np.real(doc_doc_sim))] = files[count_group]
        count_group += 1
    return all_sim
#计算这个文档属于哪一类
def  count_max(klargest_sim):
    kind_count = {}
    for key, value in klargest_sim:
        if value not in kind_count:
            kind_count[value] = 1
        else:
            kind_count[value] += 1
    largest = heapq.nlargest(1, kind_count.items(), key=lambda s: s[1])
    return largest


def get_all_key_value(dic,vec):
    space = []
    count_g = 0
    for group_vec in vec:
        group_space = []
        for doc_vec in group_vec:
            count = 0
            doc_s = {}
            for n in doc_vec:
                if n>0:
                    doc_s[dic[count_g][count]] = n
                count += 1
            group_space.append(doc_s)
        space.append(group_space)
        count_g += 1

    return space

'''

# 某测试组比对某训练组进行相似度计算
def one_test_one_train(dic_one_kind, train_vec_one_kind ,test_vec_one_kind):
    print('one_test_one_train....')
   #得到测试文件词典在训练文件中的索引
    group_group_sim = []
    for test_vec_one_doc in test_vec_one_kind:
        doc_group_sim_arr = one_doc_one_group(train_vec_one_kind, test_vec_one_doc)
        group_group_sim.append(doc_group_sim_arr) #一一对应
    return group_group_sim

# 某测试组中某一文档比对某一训练组进行相似度计算
def one_doc_one_group(train_vec_one_kind,test_vec_one_doc):
    cos_sim_arr = []
    for each_vec in train_vec_one_kind:
        fenzi = 0
        fenmu1 = 0
        fenmu2 = 0
        print(len(each_vec))
        print(len(test_vec_one_doc))
        len_doc = len(test_vec_one_doc)
        for count_i in range(len_doc):
            fenzi += test_vec_one_doc[count_i]*each_vec[count_i]
            fenmu1 += each_vec[count_i]*each_vec[count_i]
            fenmu2 += test_vec_one_doc[count_i]*test_vec_one_doc[count_i]

        fenmu = np.real(((cmath.sqrt(fenmu1))*(cmath.sqrt(fenmu2))))
        if(fenmu > 0):
            cos_sim = fenzi/((cmath.sqrt(fenmu1))*(cmath.sqrt(fenmu2)))
        else:
            cos_sim = 0
        cos_sim_arr.append(cos_sim)
    return cos_sim_arr
#test文档属于不同train类的相似度合并到一起方便求前k个最大的
def merge(group_all_key,group_group_key):
    if group_all_key:
        len_group = len(group_all_key)
        for i in range(len_group):
            group_all_key[i] = dict(group_all_key[i] , **group_group_key[i])
    else:
        group_all_key = group_group_key
    return group_all_key

#转化 为相似度为key，value为训练类别
def get_group_group_key_value(group_group_sim ,value):
    group_group_key = []
    for each_test_group in group_group_sim:
        doc_group_key = {}
        for each_doc in each_test_group:
            doc_group_key[str(each_doc)] = value
        group_group_key.append(doc_group_key)
    return group_group_key

def classify(train_dic, train_vec, train_file_kind, test_dic, test_vec, test_file_kind):
    print('classify....')
    global  count_right, count_wrong
    # 得到测试文件词典在训练文件中的索引
    index_arr = []
    for word in test_dic:
        try:
            index = train_dic.index(word)
        except ValueError:
            index = -1
        else:
            index = index
        index_arr.append(index)
    index_count = 0
    for each_test_vec in test_vec:
        classified =  doc_classified(train_dic, train_vec, train_file_kind, each_test_vec, index_arr)
        index_count =index_count+1
        original = test_file_kind[index_count]
        print('This doc is classfied to ', classified, ' ,原属于' + original )
        if classified == original:
            count_right = count_right + 1
        else:
            count_wrong = count_wrong + 1
        return count_right, count_wrong
    return count_right,count_wrong

#对一个test文档进行分类
def doc_classified(train_dic, train_vec, train_file_kind, each_test_vec, index_arr)
    #计算一个测试文档与所有文档的相似度
    cos_sim_arr = []
    for each_train_vec in train_vec:
        fenzi = 0
        fenmu1 = 0
        fenmu2 = 0
        count_i = 0
        for index in index_arr:
            if(index >= 0):
                fenzi += each_test_vec[count_i] * each_train_vec[index]
                fenmu1 += each_train_vec[index] * each_train_vec[index]
                fenmu2 += each_test_vec[count_i] * each_test_vec[count_i]
                count_i += 1
        fenmu = np.real(((cmath.sqrt(fenmu1)) * (cmath.sqrt(fenmu2))))
        if (fenmu > 0):
            cos_sim = fenzi / ((cmath.sqrt(fenmu1)) * (cmath.sqrt(fenmu2)))
        else:
            cos_sim = 0
        cos_sim_arr.append(cos_sim)
        doc_all_key = get_doc_all_key_value(cos_sim_arr ,train_file_kind)
        klargest_sim = heapq.nlargest(K, doc_all_key.items(), key=lambda s: s[0])
        largest = count_max(klargest_sim)
    return largest

#计算这个文档属于哪一类
def  count_max(klargest_sim):
    kind_count = {}
    for key, value in klargest_sim:
        if value not in kind_count:
            kind_count[value] = 1
        else:
            kind_count[value] += 1
    largest = heapq.nlargest(1, kind_count.items(), key=lambda s: s[1])
    return list(largest[0])[0]

#转化为相似度为key，value为训练类别
def get_doc_all_key_value(cos_sim_arr ,train_file_kind):
    doc_all_key = {}
    count_index = 0
    for each_sim in cos_sim_arr:
        doc_all_key[str(each_sim)] = train_file_kind[count_index]
        count_index = count_index+1
    return doc_all_key'''