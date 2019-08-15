import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.layers import Dense,LeakyReLU,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.models import Sequential,Model

import training_preprocess


#customized setting
img_size = [224,224,3]  #some pre-trained model can't accept dynamic input size
fine_tune_all_layer = True   #train last classified layer first than fine tune all layers
epochs=50
batch_size = 9

plt.switch_backend('agg')
cwd = os.getcwd()


def show_train_history(train_history,train,validation,file_name):
    fig = plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('train history')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(os.path.join(cwd,'checkpoint',file_name))
    plt.close(fig)


def training():
    classes = training_preprocess.get_classes()
    
    # load & pre-process data
    x_train_normalize,y_train_one_hot = training_preprocess.data_preprocess(img_size)
    
    # model build
    input_shape = tuple(img_size)
    net = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)


    for layer in net.layers: #freeze pre-train weight
        layer.trainable = False


    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    output = Dense(len(classes),activation='softmax')(x)

    model = Model(net.input,output)

    
    ''' #by sequential model
    model = Sequential()
    for layer in net.layers:
        model.add(layer)

    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(len(classes),activation='softmax'))
    '''
    
    print(model.summary())
    
    ### train only bottleneck first to get stable loss
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    
    # callback
    logging = TensorBoard(log_dir=os.path.join(cwd,'checkpoint'))
    filepath= os.path.join(cwd,'checkpoint','ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_weights_only=True, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    
    # Fit the model
    train_history = model.fit(x=x_train_normalize, y=y_train_one_hot,
              validation_split=0.1,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[checkpoint,logging,reduce_lr], verbose=1)
    
    # save model structure
    with open(os.path.join(cwd,'checkpoint',"mobilenet.json"), "w") as json_file:
        json_file.write(model.to_json())
    # save stage1 model weight
    model.save_weights(os.path.join(cwd,'checkpoint','trained_weights_final.h5'))
    
        
    show_train_history(train_history,'acc','val_acc','train_acc_history.png')
    show_train_history(train_history,'loss','val_loss','train_loss_history.png')
    
    ### evaluate the last model
    print('evaluate the latest model performance...')
    
    training_preprocess.augment = False
    x_train_normalize,y_train_one_hot = training_preprocess.data_preprocess(img_size)
    y_train = np.argmax(y_train_one_hot,axis = 1)
    
    train_predict = model.predict(x_train_normalize)
    train_predict_class = np.argmax(train_predict,axis=1)
    
    print('total %d training data' %len(y_train))
    print('accuracy of traing data: %f ' %(sum(y_train==train_predict_class)/len(y_train)) )
    print('confusion table of training data:')
    print(pd.crosstab(y_train,train_predict_class,rownames=['true'],colnames=['pred']))


if __name__ == '__main__':
    training()






