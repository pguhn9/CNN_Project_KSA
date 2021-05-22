import shutil
from random import seed
from random import random
import os

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def search(dirname):
    img_list = []
    for (path, dir, files) in os.walk(dirname):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.jpg':
                img_list.append(path+'/'+filename)
    return img_list

img_list = search("8-multi-class_data/train")
val_ratio = 0.05
test_ratio = 0.15

print(img_list)


data_set = './8-multi-class_data/'
subdirs = ['train/', 'val/', 'test/']
labeldirs = ['airplane/', 'car/', 'cat/', 'dog/', 'flower/', 'fruit/', 'motorbike/', 'person/']
for sub in subdirs:
    for labelsub in labeldirs:
        dir_name = data_set + sub + labelsub
        print(dir_name)
        createFolder(dir_name)

print(type(img_list))
print(img_list[0].split('/')[-1])
print(img_list[0].split('/')[-1].find('airplanae'))


for i, img in enumerate(img_list):
    print(img)
    str_dir = 'train/'
    dst_dir = 'train/'
    rnum = random()
    if rnum < val_ratio:
        dst_dir = 'val/'
    elif rnum < test_ratio:
        dst_dir = 'test/'
    if img_list[i].split('/')[-1].find('airplane') >= 0:
        dst = data_set + dst_dir + 'airplane/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    if img_list[i].split('/')[-1].find('car') >= 0:
        dst = data_set + dst_dir + 'car/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    if img_list[i].split('/')[-1].find('cat') >= 0:
        dst = data_set + dst_dir + 'cat/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    if img_list[i].split('/')[-1].find('dog') >= 0:
        dst = data_set + dst_dir + 'dog/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    if img_list[i].split('/')[-1].find('flower') >= 0:
        dst = data_set + dst_dir + 'flower/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    if img_list[i].split('/')[-1].find('fruit') >= 0:
        dst = data_set + dst_dir + 'fruit/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    if img_list[i].split('/')[-1].find('motorbike') >= 0:
        dst = data_set + dst_dir + 'motorbike/' + img_list[i].split('/')[-1]
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass
    elif img_list[i].split('/')[-1].find('person') >= 0:
        dst = data_set + dst_dir + 'person/' + img_list[i].split('/')[-1]
        print(dst)
        try:
            shutil.move(img, dst)
        except shutil.SameFileError:
            pass