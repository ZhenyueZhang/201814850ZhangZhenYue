import train as tn
import os
import math
count_right = 0
count_wrong = 0

# 得到测试数据词典
def get_all_tokens(path):
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    #file_kind = []
    all_group_tokens = []
    for file in files:  # 遍历文件夹
        file_num = 0
        one_group_tokens = []
        sFiles = os.listdir(path + "/" + file)
        for sFile in sFiles:
            if not os.path.isdir(sFile):  # 判断是否是文件夹，不是文件夹才打开
                file_num = file_num+1
                f = open(path + "/" + file + "/" + sFile, 'r', newline='\n', encoding='gb18030',errors='ignore')  # 打开文件
                str = tn.get_a_txt(f)
                unit_tokens = tn.remove_stopwords(str)
                #one_group_tokens.append(list(set(unit_tokens)))
                one_group_tokens.append(unit_tokens)
        all_group_tokens.append(one_group_tokens)
    return all_group_tokens

#dui对所有测试集文档进行分类
def all_classify(test_path,all_group_tokens,none_word_p,p_files,groups_words_p):
    count_group = 0
    global count_wrong,count_right
    kind_names = tn.get_file_kind(test_path)
    for group in all_group_tokens:
        for doc in group:
            belong_to_index = doc_classify(doc, none_word_p, p_files, groups_words_p)
            belong_to_name = kind_names[belong_to_index]
            original_belong = kind_names[count_group]
            if belong_to_name == original_belong:
                count_right += 1
            else:
                count_wrong += 1
            #print(count_right,' ',count_wrong)
            print('This doc is classfied to ', belong_to_name, ' ,original belong to ' + original_belong)
        count_group += 1
    return count_right ,count_wrong
# dui对每一个测试集文档进行分类,返回值为文档所属类别的索引0开始
def doc_classify(doc_tokens, none_word_p, p_files, groups_words_p):
    group_index = 0
    belong_p_all = {}
    for each_kind in groups_words_p:
        belong_p = 0
        for each_token in doc_tokens:
            try:
                word_p = each_kind[each_token]
            except KeyError:
                word_p = none_word_p[group_index]
            else:
                word_p = word_p
            #print(word_p)
            belong_p = belong_p+math.log10(word_p)
        belong_p_all[group_index] = belong_p+math.log10(p_files[group_index])
        #belong_p = belong_p *word_p*10000
        #print(belong_p)
    # belong_p_all[group_index] = belong_p*p_files[group_index]
        group_index += 1
    #找出最大概率值判断属于哪一类
    max_v = -10000
    re_key = 0
    for k,v in belong_p_all.items():
        if v>max_v:
            max_v = v
            re_key = k
    return re_key

