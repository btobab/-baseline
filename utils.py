import os
import matplotlib.pyplot as plt
import numpy as np
def get_path(image_path):
    files=[]
    for dir_name in os.listdir(image_path):
        for image_name in os.listdir(os.path.join(image_path,dir_name)):
            if image_name.endswith('.png') and not image_name.startswith('.'):
                files.append(os.path.join(image_path,dir_name,image_name))
    return sorted(files)

def show_images(imgs):
    #imgs是一个列表，列表里是多个tensor对象
    #定义总的方框的大小
    plt.figure(figsize=(3*len(imgs),3), dpi=80)
    for i in range(len(imgs)):
        #定义小方框
        plt.subplot(1, len(imgs), i + 1)
        #matplotlib库只能识别numpy类型的数据，tensor无法识别
        imgs[i]=imgs[i].numpy()
        #展示取出的数据
        plt.imshow(imgs[i][0],cmap="gray",aspect="auto")
        #设置坐标轴
        plt.xticks([])
        plt.yticks([])

def label_trans(label):
    label = label.transpose((2, 0, 1))
    label = label[0, :, :]
    label = np.expand_dims(label, axis=0)
    if np.mean(label)>1:
        label=label/255.
    label = label.astype("int64")

def get_test_data(test_images_path):
    test_data=[]
    for name in os.listdir(test_images_path):
        img_path=os.path.join(test_images_path,name)
        test_data.append(img_path)
    test_data=np.expand_dims(np.array(test_data),axis=1)
    return test_data