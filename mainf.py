import train as tn
import classify as cy
train_path = 'C:/data/datamining/20news-bydate/20news-bydate-train'  #文件夹目录
#test_path = 'C:/data/datamining/20news-bydate/20news-bydate-test'
test_path = 'C:/data/datamining/20news-bydate/20news-bydate-test'
none_word_p, p_files, groups_words_p = tn.get_words_p(train_path)
test_all_group_tokens = cy.get_all_tokens(test_path)

count_right, count_wrong = cy.all_classify(test_path,test_all_group_tokens,none_word_p,p_files,groups_words_p)
print('correct rate： ',count_right/(count_right+count_wrong))
print(count_right,count_wrong,(count_right+count_wrong))
