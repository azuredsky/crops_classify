from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

path = r'G:\crops_train\check'
dirnames = os.listdir(path)

label_word_dict = {}
word_label_dict = {}

num_label = LabelEncoder().fit_transform(dirnames)
print(type(num_label))
onehot_label = OneHotEncoder(sparse=False).fit_transform(np.asarray(num_label).reshape([-1,1]))
# onehot_label = list(list for each in onehot_label)

for i in range(len(dirnames)):
    label_word_dict[num_label[i]] = dirnames[i]
    word_label_dict[dirnames[i]] = list(onehot_label[i])
with open('label_word_dict.txt', 'w', encoding='utf-8') as f:
    f.write(str(label_word_dict))
with open('word_label_dict.txt', 'w', encoding='utf-8') as f:
    f.write(str(word_label_dict))




