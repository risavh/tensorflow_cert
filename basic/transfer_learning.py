import random

import tensorflow as tf
import pathlib
import os

import tensorflow.keras.losses
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential


print('TF Version: {}'.format(tf.__version__))


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
data_dir=tf.keras.utils.get_file(fname='cats_and_dogs.zip',origin=_URL,extract=True)
data_dir=pathlib.Path(data_dir)
# print(data_dir)
# print(os.path.dirname(data_dir))

data_path=os.path.join(os.path.dirname(data_dir),'cats_and_dogs_filtered')
train_dir=os.path.join(data_path,'train')
val_dir=os.path.join(data_path,'validation')

train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')

val_cats_dir=os.path.join(val_dir,'cats')
val_dogs_dir=os.path.join(val_dir,'dogs')

print('Number of Training Cats: {}, Number of Training Dogs: {}'.format(len(os.listdir(train_cats_dir)),len(os.listdir(train_dogs_dir))))
print('Number of Validation Cats: {}, Number of Validation Dogs: {}'.format(len(os.listdir(val_cats_dir)),len(os.listdir(val_dogs_dir))))

img_height=150
img_width=150
batch_size=32

train_ds=image_dataset_from_directory(train_dir, batch_size=batch_size, image_size=(img_height, img_width), shuffle=True, seed=123)
val_ds=image_dataset_from_directory(val_dir, batch_size=batch_size, image_size=(img_height, img_width), shuffle=True, seed=123)


class_names=train_ds.class_names
print(class_names)

fig,ax=plt.subplots(2,5,figsize=(15,7))
for imgs,labels in train_ds.take(1):
    for i in range(2):
        for j in range(5):
            img_num=random.randint(0,batch_size)
            ax[i,j].imshow(imgs[img_num].numpy().astype('uint8'))
            ax[i,j].set_title(class_names[labels[img_num]])
            ax[i, j].grid('off')
plt.show()


## Model Building

model=Sequential()
model.add(tensorflow.keras.layers.experimental.preprocessing.Rescaling((1./255),input_shape=(img_height,img_height,3)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(2))
model.summary()


model.compile(optimizer='adam',loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

history=model.fit(train_ds,epochs=10,validation_data=val_ds)


fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('accuracy')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()
plt.show()


