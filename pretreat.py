import os
import numpy as np

Width, Height = 224, 224
valpath = r'G:\crops_train\dataset\val'
testpath = r'G:\crops_train\dataset\test'
trainpath = r'G:\crops_train\dataset\train'


def generate_arrays_from_file_1(trainpath,set_len=21600,file_nums=6,has_remainder=0,batch_size=32):

    '''
    :param trainsetpath: 训练集路径
    :param set_len: 训练文件的图片数量
    :param file_nums: 训练文件数量
    :param has_remainder: 是否有，即has_remainder = 0 if set_len % batch_size == 0 else 1
    :param batch_size:
    :return:
    '''

    cnt = 0
    pos = 0
    inputs = None
    labels = None
    while 1:
        if cnt % (set_len//batch_size+has_remainder) == 0:
            pos = 0
            seq = cnt//(set_len//batch_size+has_remainder) % file_nums
            del inputs,labels
            inputs = np.load(os.path.join(trainpath, 'inputs'+str(seq)+'.npy'))
            labels = np.load(os.path.join(trainpath, 'labels'+str(seq)+'.npy'))
        # print('ok')
        start = pos*batch_size
        end = min((pos+1)*batch_size, set_len-1)
        batch_inputs = inputs[start:end]
        batch_labels = labels[start:end]
        pos += 1
        cnt += 1

        yield (batch_inputs,batch_labels)


