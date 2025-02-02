import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from keras.callbacks import ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.models import load_model

import model_structure
import training_preprocess

plt.switch_backend('agg')

#customized setting
img_size = [85,70,3]
epochs = 50
batch_size = 9
optimizer = 'adam'

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
    
    # load & pre-process data
    classes = training_preprocess.get_classes()
    x_train_normalize,y_train_one_hot = training_preprocess.data_preprocess(img_size)
    
    # model build
    input_shape = tuple(img_size)
    model = model_structure.model_body(input_shape,num_classes=len(classes))
    
    print(model.summary())
    
    ### training ###
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    # checkpoint
    #filepath= os.path.join(cwd,'checkpoint','best_epoches_model.h5')
    filepath= os.path.join(cwd,'checkpoint','ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    logging = TensorBoard(log_dir=os.path.join(cwd,'checkpoint'))
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_weights_only=True, save_best_only=True, mode='min')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    callbacks_list = [checkpoint,logging]
    
    # Fit the model
    train_history = model.fit(x=x_train_normalize,
                              y=y_train_one_hot,
                              validation_split=0.1,epochs=epochs,batch_size=batch_size,
                              callbacks=callbacks_list, verbose=1)
    
    # save model structure
    with open(os.path.join(cwd,'checkpoint',"model_structure.json"), "w") as json_file:
        json_file.write(model.to_json())
    
    # save final epoch model weight
    model.save_weights(os.path.join(cwd,'checkpoint','trained_weights_final_epoch.h5'))
    
    
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






