# 该文件中的函数主要是对数据进行预处理，然后得到词典和向量空间，并将它们写入本地
from nltk.corpus import stopwords
import re
import nltk
import os
#import math
lower = 0

# 得到训练数据的词频概率
def get_words_p(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    #file_kind = []
    files_num = []
    p_files = []
    num_all_files = 0
    all_group_tokens = []
    for file in files:  # 遍历文件夹
        file_num = 0
        one_group_tokens = []
        sFiles = os.listdir(path + "/" + file)
        for sFile in sFiles:
            if not os.path.isdir(sFile):  # 判断是否是文件夹，不是文件夹才打开
                file_num = file_num+1
                f = open(path + "/" + file + "/" + sFile, 'r', newline='\n', encoding='gb18030',errors='ignore')  # 打开文件
                str = get_a_txt(f)
                unit_tokens = remove_stopwords(str)
                #file_kind.append(file)
            one_group_tokens.append(unit_tokens)
        num_all_files += file_num
        files_num.append(file_num)
        all_group_tokens.append(one_group_tokens)
    for each in files_num:
        p_files.append(each/num_all_files)

    none_word_p,groups_words_p = get_groups_p(all_group_tokens)
    return none_word_p, p_files, groups_words_p

# 训练集得词典单词概率
def get_groups_p(all_group_tokens):
    global lower
    # 去低频词
    groups_words_p = []
    words_number = 0
    none_word_p = []
    for one_group_tokens in all_group_tokens:
        word_dict = {}
        for doc_tokens in one_group_tokens:
            for unit_token in doc_tokens:
                if unit_token not in word_dict:
                    word_dict[unit_token] = 1
                else:
                    word_dict[unit_token] += 1
                words_number += 1
        filt_word_dict = {}
        words_count = 0
        for k,v in word_dict.items():
            if v>lower:
                filt_word_dict[k] = v
                words_count += v
        print('得到词典词频...单词个数为', len(filt_word_dict))
        group_words_p = {}
        for k,v in filt_word_dict.items():
            group_words_p[k] = (v+1)/(words_count+len(filt_word_dict))
        groups_words_p.append(group_words_p)
        none_word_p.append(1/(words_count+len(filt_word_dict)))
    return none_word_p, groups_words_p

# 去非英文字符
def remove_punctuation(line):
    rule = re.compile(u"[^a-zA-Z\s]")
    line = rule.sub('', line)
    return line


# 去停用词
def remove_stopwords(line):
    tokens = nltk.word_tokenize(line)
    noPieTokens = [word for word in tokens if '\'' not in word]  # 去掉带有‘ 的，如's ，n't
    noSymbolTokens = [remove_punctuation(word) for word in noPieTokens]  # 去掉特殊符号
    lowerTokens = [word.lower() for word in noSymbolTokens]  # 小写
    swords = stopwords.words('english')
    swords = swords + ['']
    filtered_words = [word for word in lowerTokens if word not in swords]  # 去除包含空的停用词
    return filtered_words


# 读取一个文档
def get_a_txt(f):
    iter_f = iter(f);  # 创建迭代器
    str = ""
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        str = str + line
    return str


# 获得所有类文件夹的名称
def get_file_kind(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    return files

'''# 向量改为实数
def get_real(vec):
    new_vec = []
    for each_vec in vec:
        vec_row = []
        for each in each_vec:
            vec_row.append(np.real(each))
            new_vec.append(vec_row)
    return new_vec
'''