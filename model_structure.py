from functools import reduce
from keras.layers import Dense,Conv2D,MaxPooling2D,initializers,LeakyReLU,Input,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model



def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def conv_block(**kwargs):
    return compose(
        Conv2D(**kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def model_body(inputs,num_classes):
    
    initial = initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=100)
    
    x0 = Input(inputs)
    x = conv_block(filters = 32,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x0)
    x = conv_block(filters = 32,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = conv_block(filters = 64,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 64,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
     
    x = conv_block(filters = 48,kernel_size=(1,1),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 128,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 128,kernel_size=(1,1),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 128,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = conv_block(filters = 64,kernel_size=(1,1),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 128,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 64,kernel_size=(1,1),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 128,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    
    x = conv_block(filters = 256,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = conv_block(filters = 256,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    
    x = Conv2D(filters = 256,kernel_size=(3,3),padding = 'same',kernel_initializer=initial)(x)
    x = GlobalAveragePooling2D()(x) # GAP (Flatten)
    
    predictions = Dense(num_classes, activation="softmax",kernel_initializer=initial)(x)
    
    return Model(x0, predictions) 


    
    
