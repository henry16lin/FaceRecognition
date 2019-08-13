import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import model_structure
import training_preprocess


#customized setting
img_size = [85,70,3]
epochs = 50
batch_size = 9
optimizer = 'adam'
num_classes = 3

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
    
    # load & pre-process data
    x_train_normalize,y_train_one_hot = training_preprocess.data_preprocess(img_size)
    
    # model build
    input_shape = tuple(img_size)
    model = model_structure.model_body(input_shape,num_classes=num_classes)
    
    print(model.summary())
    
    ### training ###
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    # checkpoint
    #filepath= os.path.join(cwd,'checkpoint','best_epoches_model.h5')
    filepath= os.path.join(cwd,'checkpoint','ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    logging = TensorBoard(log_dir=os.path.join(cwd,'checkpoint'))
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_weights_only=True, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    callbacks_list = [checkpoint,logging,reduce_lr]
    
    # Fit the model
    train_history = model.fit(x=x_train_normalize,
                              y=y_train_one_hot,
                              validation_split=0.1,epochs=epochs,batch_size=batch_size,
                              callbacks=callbacks_list, verbose=1)
    
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






