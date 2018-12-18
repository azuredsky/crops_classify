import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import gc
LOC = list(np.zeros((216,),dtype=int))

path = r'G:\crops_train\check'

Width, Height = 224,224

trainpath = r'G:\crops_train\dataset\train'
testpath = r'G:\crops_train\dataset\test'
valpath = r'G:\crops_train\dataset\val'


def img_pretreat(file):

    row_img = Image.open(file)
    img = row_img.resize((Width, Height))
    points = np.asanyarray(img, dtype=np.float32)
    points = points * 1. / 255
    points = np.reshape(points, [Width, Height, 3])
    return points


def img2arr_train(path,step):

    if step == 6:
        NUM = 50
    else:
        NUM = 100

    with open('word_label_dict.txt', 'r', encoding='utf-8') as f:
        js = eval(f.read())

    img_data_list = []
    row_label = []

    dirnames = os.listdir(path)
    for cls,dirname in enumerate(tqdm(dirnames)):
        dir = os.path.join(path,dirname)
        start = LOC[cls]
        num = 0
        for parent,_,filenames in os.walk(dir):
            for filename in filenames[start:]:
                LOC[cls] += 1
                try:

                    file = os.path.join(parent,filename)
                    i = random.randint(0, len(img_data_list))
                    img_data = img_pretreat(file)
                    img_data_list.insert(i, img_data)
                    row_label.insert(i,js[dirname])
                    num += 1
                    # print(num)
                    if num > NUM:
                        break
                except Exception as e:
                    # print(e)
                    continue
        print(LOC[cls])

    inputs = np.array(img_data_list)
    labels = np.array(row_label)
    save_path = ''
    if step < 6:
        save_path = trainpath
    if step == 6:
        save_path = valpath
    if step == 7:
        save_path = testpath

    np.save(os.path.join(save_path,'inputs'+str(step)+'.npy'),inputs)
    np.save(os.path.join(save_path,'labels'+str(step)+'.npy'),labels)
    del inputs,labels
    gc.collect()


if __name__ == "__main__":

    for step in range(0,8):
        img2arr_train(path,step)