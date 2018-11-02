#该文件中的函数主要是对数据进行预处理，然后得到词典和向量空间，并将它们写入本地
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import math
upper = 0.005#高频词上限，即某词出现的次数占所有词的百分比
lower = 2 #低频词下限，即某词出现次数

#去非英文字符
def remove_punctuation(line):
    rule = re.compile(u"[^a-zA-Z\s]")
    line = rule.sub('', line)
    return line

#去停用词
def remove_stopwords(line):
    tokens = nltk.word_tokenize(line)
    noPieTokens = [word for word in tokens if '\'' not in word]#去掉带有‘ 的，如's ，n't
    noSymbolTokens = [remove_punctuation(word) for word in noPieTokens ]#去掉特殊符号
    lowerTokens = [word.lower() for word in noSymbolTokens]#小写
    swords = stopwords.words('english')
    swords = swords+['']
    filtered_words = [word for word in lowerTokens if word not in swords]#去除包含空的停用词
    return filtered_words

#读取一个文档
def get_a_txt(f):
    iter_f = iter(f);  # 创建迭代器
    str = ""
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        str = str + line
    return str

#得到训练数据的词典和向量空间
def get_dic_vec(path, groups_dic,all_group_dic):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    file_kind = []
    all_group_tokens = []
    for file in files:  # 遍历文件夹
        one_group_tokens = []
        sFiles = os.listdir(path + "/" + file)
        for sFile in sFiles:
            if not os.path.isdir(sFile):  # 判断是否是文件夹，不是文件夹才打开
                f = open(path + "/" + file + "/" + sFile, 'r',newline='\n', encoding='gb18030', errors='ignore')  # 打开文件
                str = get_a_txt(f)
                unit_tokens = remove_stopwords(str)
                file_kind.append(file)
            one_group_tokens.append(unit_tokens)
        all_group_tokens.append(one_group_tokens)
    global  upper, lower
    if 'test' in path :
        vec, dic, all_group_dic = get_test_tf_idf(all_group_tokens,groups_dic,all_group_dic)
    else:
        vec, dic, all_group_dic = get_tf_idf(all_group_tokens)
    return dic,vec,all_group_dic

import numpy as np
#向量改为实数
def get_real(vec ):
    new_vec = []
    for each_vec in vec:
        vec_row = []
        for each in each_vec:
            vec_row.append(np.real(each))
            new_vec.append(vec_row)
    return  new_vec

# 训练集得词典和tf*idf
def get_tf_idf(all_group_tokens):
    #去低频词
    word_dict = {}
    words_number = 0
    for one_group_tokens in all_group_tokens:
        for doc_tokens in one_group_tokens:
            for unit_token in list(set(doc_tokens)):
                if unit_token not in word_dict:
                 word_dict[unit_token] = 1
                else:
                  word_dict[unit_token] += 1
                words_number += 1
    print('未去重前词的数量 ', words_number)
    all_group_dic = [key for key, value in word_dict.items() if (value > lower) & (value/words_number < upper)]  # 找出词频满足条件词，即词典
    print('得到大词典...个数为',len(all_group_dic))
    # 计算各个组的无重词典
    groups_dic_mul = []
    all_tf = []
    for one_group_tokens in all_group_tokens:
        group_tokens = []
        for doc_tokens in one_group_tokens:
            group_tokens = group_tokens+doc_tokens
        nom_group_tokens = list(set(group_tokens))
        group_dic = [word for word in nom_group_tokens if word in all_group_dic ]
        groups_dic_mul.append(group_dic)
        print('完成一个小词典...个数为',len(group_dic))
    print('得到各个组小词典...')
    #groups_dic = remove_common(groups_dic_mul)
    groups_dic = groups_dic_mul
    count = 0
    for group_dic in groups_dic:
        one_group_tf = get_tf(group_dic,all_group_tokens[count])
        count += 1
        all_tf.append(one_group_tf)
    print('得到tf...')
    idf = get_idf(all_group_dic, groups_dic, all_group_tokens)
    print('得到idf...')
    tf_idf = tf_idf_cal(all_tf, groups_dic, idf)
    print('得到tf-idf...')
    return  tf_idf,groups_dic,all_group_dic

# 测试文档得到tf*idf
def get_test_tf_idf(all_group_tokens,groups_dic,all_group_dic):

    all_tf = []
    count = 0
    for group_dic in groups_dic:
        one_group_tf = get_tf(group_dic, all_group_tokens[count])
        count += 1
        all_tf.append(one_group_tf)
    print('得到tf...')
    idf = get_idf(all_group_dic, groups_dic, all_group_tokens)
    print('得到idf...')
    tf_idf = tf_idf_cal(all_tf, groups_dic, idf)
    print('得到tf-idf...')
    return  tf_idf,groups_dic,all_group_dic

#计算tf*idf
def tf_idf_cal(all_tf, groups_dic, idf):
    count_dic_group = 0
    for each_group_dic in groups_dic:
        count = 0
        for word in each_group_dic:
            try:
                idf_value = idf[word]
            except KeyError:
                idf_value = 0
            else:
                idf_value = idf_value
            group_tf = all_tf[count_dic_group]
            for doc_tf in group_tf:
                doc_tf[count] = round(float(doc_tf[count] )* float(idf_value)*100,4)
            count = count + 1
        count_dic_group += 1
    return  all_tf

#得出tf
def get_tf(group_dic,one_group_tokens):
    #统计每个文档词频
    group_word_count = []
    for each_doc in one_group_tokens:
        doc_word_count = {}
        for each_token in each_doc:
            if each_token not in doc_word_count:
                doc_word_count[each_token] = 1
            else:
                doc_word_count[each_token] += 1
        group_word_count.append(doc_word_count)
    #计算出现最多单词数
    max_count = []
    for each_doc_count in group_word_count:
        max_num = 0
        for key,value in each_doc_count.items():
            if max_num < each_doc_count[key]:
                max_num = each_doc_count[key]
        max_count.append(max_num)
    #计算tf
    c_doc = 0
    group_tf = []
    for each_doc_count in group_word_count:
        doc_tf = []

        fenmu = len(doc_word_count)
        for word in group_dic:
            try:
                frequency = each_doc_count[word]
            except KeyError:
                frequency = 0
            else:
                frequency = frequency
            word_tf = frequency / max_count[c_doc]
            doc_tf.append(word_tf)
        c_doc += 1
        group_tf .append(doc_tf)
    return group_tf

#计算idf
def get_idf(all_group_dic,groups_dic, all_group_tokens):
    len_all_doc = 0
    idf = {}
    for each_group in all_group_tokens:
        len_all_doc = len_all_doc + len(each_group)
    count_group = 0
    for each_group in all_group_tokens:
        for each_doc in each_group:
            doc = list(set(each_doc))
            for word in doc:
                if word in groups_dic[count_group]:
                    if word not in idf:
                        idf[word] = 1
                    else:
                        idf[word] += 1
        count_group = count_group + 1

    for key,value in idf.items():
        idf[key] = math.log(len_all_doc/(value+1))
    return idf

#从本地读入向量空间
def get_in_vec(vec_path,file_path):
    file_kind = get_file_kind(file_path)
    count = 0
    all_vec = []
    files = os.listdir(vec_path)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            group_vec = []
            f = open(vec_path+'/'+file)
            iter_f = iter(f);  # 创建迭代器
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                doc_vec = []
                sp_line = line.split(' ')
                for each in sp_line:
                    doc_vec.append(float(each))
                group_vec.append(doc_vec)

            all_vec.append(group_vec)
            count += 1
    return all_vec

#向量空间写入本地
def out_vec(vec_path, groups_vec):
    le = ['a','b','c','d','e ','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t']
    group_count = 0
    for group_vec in groups_vec:
        f = open(vec_path+le[group_count]+'.txt', 'w')
        count_line = 0
        len_group_vec = len(group_vec) - 1
        for doc_vec in group_vec:
            count = 0
            len_doc_vec = len(doc_vec)-1
            for each in doc_vec:
                f.write(str(each))
                if(count<len_doc_vec):
                    f.write(' ')
                    count = count+1
            if (count_line < len_group_vec):
                f.write('\n')
                count_line = count_line+1
        group_count += 1
    return 1

#词典写入本地
def out_dic(dic_path, groups_dic):
    f = open(dic_path,'w')
    count_group = 0
    len_groups = len(groups_dic)-1
    for each_dic in groups_dic:
        count_word = 0
        len_dic = len(each_dic)-1
        for word in each_dic:
            f.write(word)
            if(count_word<len_dic):
                f.write(' ')
            count_word += 1
        if count_group<len_groups:
            f.write('\n')
            count_group += 1
    return 1

#读入词典
def get_in_dic(dic_path,file_path):
    f = open(dic_path)
    iter_f = iter(f);  # 创建迭代器
    groups_dic = []
    count  = 0
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        sp_line = line.split(' ')
        group_dic = []
        for each in sp_line:
            group_dic.append(each)
        groups_dic.append(group_dic)
        count +=1
    return groups_dic

#获得所有类文件夹的名称
def get_file_kind(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    return files

#若每个类的文件夹的小词典与其它类文件夹的小词典有共同词，则去掉
def remove_common(groups_dic):
    return_groups = []
    len_dic  = len(groups_dic)
    for i in range(len_dic):
        listA = groups_dic[i]
        return_groups.append(remove_common_group( listA , groups_dic, i))
    return return_groups

#一个小词典去掉与其他词典共同拥有的词
def remove_common_group( listA , groups_dic, i):
    len_dic = len(groups_dic)
    for j in range(len_dic):
        if i != j:
            list_re = [word for word in listA if word not in groups_dic[j]]
    print('去共同词后的小词典词数： ', len(list_re))
    return list_re


'''
#文档去掉高频词和低频词，upper是百分比，lower是整数
def remove_nosense_words(all_group_tokens, upper, lower):
    #统计单词出现次数
    word_dict = {}
    count=0 #记录总单词数
    for unit_tokens in all_group_tokens:
        for word in unit_tokens:
            count += 1
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    print('nosense_be')
    no_sense_words = [key for key, value in word_dict.items() if  value < lower]#找出词频高于上界和低于下界的词
    print('nosense')
    return_tokens = []  # 去掉无用词的文档单词
    count = 0
    for unit_tokens in all_group_tokens:
        print(count)
        count=count+1
        unit_words = [word for word in unit_tokens if word not in no_sense_words]
        return_tokens.append(unit_words)
    return return_tokens
    
# 得词典和tf-idf
def tf_idf(all_tokens):
    group_txt = []
    for unit_tokens in all_tokens:
        single_txt = (' ').join(unit_tokens)
        group_txt.append(single_txt)
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(group_txt)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    # 将词频矩阵X统计成TF-IDF值
    tf_idf = transformer.fit_transform(X)
    # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
    return word, tf_idf.toarray()

'''