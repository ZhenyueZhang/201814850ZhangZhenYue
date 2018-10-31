import classifyFunction as cf
import definFunction as df
'''
train_path = 'C:/data/datamining/20news-bydate-train'  #文件夹目录
test_path = 'C:/data/datamining/20news-bydate-test'#'C:/data/datamining/20news-bydate-test'
train_vec_path = 'C:/data/train_vec.txt'
test_vec_path = 'C:/data/test_vec.txt'
train_dic_path = 'C:/data/train_dic.txt'
test_dic_path = 'C:/data/test_dic.txt'''
train_path = 'C:/data/datamining/20news-bydate-train'  #文件夹目录
test_path = 'C:/data/datamining/20news-bydate-test'
train_vec_path = 'C:/data/travec/train_vec'
test_vec_path = 'C:/data/testvec/test_vec'
train_dic_path = 'C:/data/train_dic.txt'
test_dic_path = 'C:/data/test_dic.txt'
read_in_train_vec_path = 'C:/data/travec'
read_in_test_vec_path = 'C:/data/testvec'



test_vec = df.get_in_vec(read_in_test_vec_path, test_path)
dic = df.get_in_dic(train_dic_path, train_path)
train_vec = df.get_in_vec(read_in_train_vec_path, train_path)
files = df.get_file_kind(train_path)
count_right,count_wrong = cf.classify( train_vec, dic, test_vec,files)

print(count_right)
print(count_wrong)
print(count_right/(count_right+count_wrong))
'''

test_dic,test_vec,test_file_kind = df.get_train_dic_vec(test_path)
df.out_dic(test_dic_path,test_dic)
df.out_vec(test_vec_path,df.get_real(test_vec))
test_dic,test_vec, test_file_kind = df.get_dic_vec(test_path)
df.out_dic(test_dic_path,test_dic)
df.out_vec(test_vec_path,test_vec)
train_dic,train_vec, train_file_kind = df.get_dic_vec(train_path)
df.out_dic(train_dic_path,train_dic)
df.out_vec(train_vec_path,train_vec)


groups_dic = []
all_group_dic = []
train_dic,train_vec, all_group_dic = df.get_dic_vec(train_path,groups_dic,all_group_dic)
df.out_dic(train_dic_path,train_dic)
df.out_vec(train_vec_path,train_vec)
test_dic,test_vec, all_group_dic = df.get_dic_vec(test_path,train_dic,all_group_dic)
df.out_vec(test_vec_path,test_vec)

groups_dic = []
all_group_dic = []
train_dic,train_vec, all_group_dic = df.get_dic_vec(train_path,groups_dic,all_group_dic)
df.out_dic(train_dic_path,train_dic)
df.out_vec(train_vec_path,train_vec)
test_dic,test_vec, all_group_dic = df.get_dic_vec(test_path,train_dic,all_group_dic)
df.out_vec(test_vec_path,test_vec)
'''
