import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import training_preprocess


#customized setting
img_size = [85,70,3]
fine_tune_all_layer = True   #train last classified layer first than fine tune all layers
epoch_first , epoch_all = 30 , 20
batch_size_bottleneck , batch_size_all = 16 , 5
optimizer = 'adam'


cwd = os.getcwd()


def show_train_history(train_history,train,validation):
    fig = plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('train history')
    plt.xlabel('epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.savefig(os.path.join(cwd,'checkpoint','training_history.png'))
    plt.close(fig)


def training():
    classes = training_preprocess.get_classes()
    
    # load & pre-process data
    x_train_normalize,y_train_one_hot = training_preprocess.data_preprocess(img_size)
    
    # model build
    input_shape = tuple(img_size)
    net = VGG16(include_top=False, weights='imagenet', input_tensor=None,input_shape=input_shape)
    
    for layer in net.layers:
        layer.trainable = False
        
    x = net.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    output_layer = Dense(len(classes), activation='softmax', name='softmax')(x)
    
    model = Model(inputs=net.input, outputs=output_layer)

    print(model.summary())
    
    ### train only bottleneck first to get stable loss
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    # checkpoint
    filepath= os.path.join(cwd,'checkpoint','best_epoches_model.h5')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    # Fit the model
    train_history = model.fit_generator(x=x_train_normalize,
                                        y=y_train_one_hot,
                                        validation_split=0.1,epochs=epoch_first,batch_size=batch_size_bottleneck,
                                        callbacks=callbacks_list, verbose=1)
    model.save_weights(os.path.join(cwd,'checkpoint','trained_weights_stage_1.h5'))
    
    if fine_tune_all_layer:
        ## unfreeze and continue training, to fine-tune (need more computational ability)
        print('unfreeze all of the layers')
        for layer in model.layers:
            layer.trainable = True
        
        train_history = model.fit_generator(x=x_train_normalize,
                                            y=y_train_one_hot,
                                            validation_split=0.1, batch_size=batch_size_all,
                                            epoch = epoch_all,
                                            initial_epoch=epoch_first,
                                            callbacks=callbacks_list, verbose=1)
        
        model.save_weights(os.path.join(cwd,'checkpoint','trained_final_weights.h5'))
        
    show_train_history(train_history,'acc','val_acc')
    show_train_history(train_history,'loss','val_loss')
    
    
    
    ### evaluate the last model
    print('evaluate model performance...')
    best_model = load_model(filepath)
    
    training_preprocess.augment = False
    x_train_normalize,y_train_one_hot = training_preprocess.data_preprocess(img_size)
    y_train = np.argmax(y_train_one_hot,axis = 1)
    
    train_predict = best_model.predict(x_train_normalize)
    train_predict_class = np.argmax(train_predict,axis=1)
    
    print('total %d training data' %len(y_train))
    print('accuracy of traing data: %f ' %(sum(y_train==train_predict_class)/len(y_train)) )
    print('confusion table of training data:')
    print(pd.crosstab(y_train,train_predict_class,rownames=['true'],colnames=['pred']))


if __name__ == '__main__':
    training()






