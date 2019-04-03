import os
import numpy as np
from scipy import misc
from keras.utils import np_utils
import skimage as sk
from scipy import ndarray
from skimage import util

cwd = os.getcwd()
augment = True


def get_classes():
    class_txt = open('classes.txt','r')
    classes = class_txt.read().split(',')
    
    train_folder = os.path.join(cwd,'training_data')
    folder_classes = os.listdir(train_folder)
    
    if len([c for c in classes if c in folder_classes]) == len(folder_classes):
        return classes
    else:
        return 'classes error...check classes.txt and training_data folder!'


def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array,var = 0.005)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]


def data_preprocess(img_size):
    
    # load data
    classes = get_classes()
    for i in range(len(classes)):
        current_class_path = os.path.join(cwd,'training_data',classes[i])
        current_data_list = os.listdir(current_class_path)
        
        np.random.shuffle(current_data_list) #shuffle the data list
        
        # read img
        current_img = []
        for j in range(len(current_data_list)):
            tmp1 = misc.imread(os.path.join(current_class_path , current_data_list[j]))
            
            if augment: ## data augment: flip & add noise
                tmp2 = horizontal_flip(tmp1)
                tmp3 = random_noise(tmp1)
                tmp4 = random_noise(tmp2)
                
                # resize
                tmp1 = misc.imresize(tmp1,img_size)
                tmp2 = misc.imresize(tmp2,img_size)
                tmp3 = misc.imresize(tmp3,img_size)
                tmp4 = misc.imresize(tmp4,img_size)
                #plt.imshow(tmp4)
                
                current_img.append(tmp1)
                current_img.append(tmp2)
                current_img.append(tmp3)
                current_img.append(tmp4)
            else:
                tmp1 = misc.imresize(tmp1,img_size)
                current_img.append(tmp1)
            
            current_cat = np.repeat(i,len(current_img))
        
        if i ==0:
            img = current_img
            cat = current_cat
        else:
            img = np.concatenate((img,current_img), axis=0).astype('float32') 
            cat = np.concatenate((cat,current_cat), axis=0).astype('float32') 
        
    # normalize & transform to 4D array
    #x_train4D = img.reshape(img.shape[0],img_size).astype('float32')
    x_train_normalize = img / 255
    
    # one-hot-encoding
    y_train_one_hot = np_utils.to_categorical(cat)
    
    return x_train_normalize,y_train_one_hot
            




  

