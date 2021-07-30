import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)




(X_train,y_train),(X_test,y_test)=mnist.load_data()

print("X_train Shape: {}, y_train Shape:{}".format(X_train.shape,y_train.shape))
print("X_test Shape: {}, y_test Shape:{}".format(X_test.shape,y_test.shape))

## Show Imag1

fig,ax=plt.subplots(2,5,figsize=(12,5))

for i in range(2):
  for j in range(5):
    #print(i,j)
    val=np.random.randint(0,10000)
    ax[i,j].imshow(X_train[val],cmap='gray' )
    ax[i,j].set_title(y_train[val],fontsize=12, )
plt.show()

## Normalizing data

X_train,X_test=X_train/255.0,X_test/255.0

## Build Model

model =Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.summary()

## Callbcaks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc')>0.96:
            print('You better stop, too good uffffff!!')
            self.model.stop_training=True


callbacks=myCallback()
## Model compile

model.compile(optimizer='adam',loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])

## Model Fit
history=model.fit(X_train,y_train,epochs=5,callbacks=[callbacks])

res=model.evaluate(X_test,y_test)
print("_"*50)
print("Test Accuracy for MNIST dataset: {} %".format(np.round(res[1]*100)))