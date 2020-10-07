import numpy as np
from tensorflow import keras
from functools import partial

def classifier(train_X, train_Y, 
               l2, dropout, lr, 
               epochs=20, 
               batch_size=32,
               seed=42):
    
    np.random.seed(seed)
    regularizer = keras.regularizers.l2(l2)
    CustomGRU = partial(keras.layers.GRU,
                        kernel_regularizer=regularizer,
                        dropout=dropout,recurrent_dropout=dropout)
    
    model = keras.models.Sequential([
                              CustomGRU(16,return_sequences=True,input_shape=[None, train_X.shape[-1]]),
                              CustomGRU(16,return_sequences=True),
                              CustomGRU(16),
                              keras.layers.Dense(1,activation='sigmoid')
                              ])
    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['acc'])
    model.fit(train_X,train_Y,epochs=epochs, validation_split=0.2,batch_size=batch_size,verbose=0)

    return model

