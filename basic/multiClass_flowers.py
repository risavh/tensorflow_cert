import tensorflow as tf
import tensorflow
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout


print(tf.__version__)


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tensorflow.keras.utils.get_file(fname='flower_photos',origin=dataset_url,untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)

batch_size = 32
img_height = 150
img_width = 150

train_ds=tensorflow.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                                     batch_size=batch_size,
                                                                     image_size=(img_height,img_width),
                                                                     seed=123,
                                                                     validation_split=0.2
                                                                     , subset='training')

valid_ds=tensorflow.keras.preprocessing.image_dataset_from_directory(directory=data_dir,
                                                                     batch_size=batch_size,
                                                                     image_size=(img_height,img_width),
                                                                     seed=123,
                                                                     validation_split=0.2
                                                                     , subset='validation')

class_names=train_ds.class_names
num_class=len(class_names)
print(class_names,num_class)

for img_fl,labl_fl in train_ds.take(1):
    fig,ax=plt.subplots(3,3,figsize=(15,7))
    cnt=0
    for i in range(3):
        for j in range(3):
            ax[i,j].imshow(img_fl[cnt].numpy().astype('uint8'))
            ax[i, j].set_title(class_names[labl_fl[cnt]])
            cnt+=1
    plt.tight_layout()
    plt.show()


## Model Building
model=tensorflow.keras.models.Sequential()
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(img_height,img_width,3)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(num_class))
model.summary()

model.compile(optimizer='adam',loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

history=model.fit(train_ds,epochs=10,validation_data=valid_ds)

model.evaluate(valid_ds)

fig,ax=plt.subplots(1,2,figsize=(12,5))
epochs=range(len(history.history['accuracy']))

ax[0].plot(epochs,history.history['accuracy'],label='train',lw=2,marker='o',color='tomato')
ax[0].plot(epochs,history.history['val_accuracy'],label='val',lw=2,marker='*',color='teal')
ax[0].set_title('Accuracy')
ax[0].legend()


ax[1].plot(epochs,history.history['loss'],label='train',lw=2,marker='o',color='tomato')
ax[1].plot(epochs,history.history['val_loss'],label='val',lw=2,marker='o',color='teal')
ax[1].set_title('Loss')
ax[1].legend()
plt.show()


