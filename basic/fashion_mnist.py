import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout, Conv2D,MaxPool2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np


(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape,y_train.shape))
print('X_test shape: {}, y_test shape: {}'.format(X_test.shape,y_test.shape))

label_mapping={0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
print(label_mapping)


fig,ax=plt.subplots(2,5,figsize=(12,5))
for i in range(2):
  for j in range(5):
    #print(i,j)
    val=np.random.randint(0,10000)
    ax[i,j].imshow(X_train[val],cmap='gray' )
    ax[i,j].set_title(label_mapping[y_train[val]],fontsize=12, )
plt.show()

## Normalizing data
X_train,X_test=X_train/255.0,X_test/255.0

## Reshape data to input into Conv2D layer
X_train=X_train.reshape(60000, 28, 28,1)
X_test=X_test.reshape(10000, 28, 28,1)

## Callback
class myCallBacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc')>0.91:
            print("Too good to be true, kill it!\n")
            self.model.stop_trainig=True

callbacks=myCallBacks()


## Model Building

model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10))
model.summary()


model.compile(optimizer='adam',loss=SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])

model.fit(X_train,y_train,epochs=5,callbacks=[callbacks])

model.evaluate(X_test,y_test)

## Feature Extractor (https://keras.io/guides/sequential_model/)

feature_extractor=tf.keras.Model(inputs=model.inputs,outputs=[layer.output for layer in model.layers])

