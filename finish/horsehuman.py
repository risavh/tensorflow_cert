import os
import zipfile
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow.keras.preprocessing.image

print("TF Version: {}".format(tf.version.VERSION))

zipfile_training = r"D:\works\koglu\datasets\horse_human\horse-or-human.zip"
zipfile_validation = r"D:\works\koglu\datasets\horse_human\validation-horse-or-human.zip"

zipref = zipfile.ZipFile(zipfile_training, "r")
zipref.extractall(r"D:\works\koglu\datasets\horse_human\training")

zipref = zipfile.ZipFile(zipfile_validation, "r")
zipref.extractall(r"D:\works\koglu\datasets\horse_human\validation")

zipref.close()

train_dir = r"D:\works\koglu\datasets\horse_human\training"
valid_dir = r"D:\works\koglu\datasets\horse_human\validation"


def visualize_data(labels, dir, title):
    """
    Uses matplotlib to visualize PIL images
    :param labels (list) :  list of labels eg: ['cats','dogs']
    :param dir: path of files with labels as subfolders
    :param title: title of the plot
    :return: visualize 5 images from each label randomly
    """
    # labels=['horses','humans']
    num_class = len(labels)
    # train_dir="/tmp/horse-or-human/"
    fig, ax = plt.subplots(num_class, 5, figsize=(15, 7))
    for i in range(num_class):
        for j in range(5):
            img_list = (os.listdir(os.path.join(dir, labels[i])))
            img = img_list[random.randint(0, len(img_list))]
            ax[i, j].imshow(mpimg.imread(os.path.join(dir, labels[i], img)))
            ax[i, j].set_title(img)
            ax[i, j].axis('off')
    plt.suptitle(title)
    plt.show()


## Example
labels = ['horses', 'humans']
dir = train_dir
title = 'Training Data'
#visualize_data(labels, dir, title)

train_generator= tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_generator= tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_datagen = train_generator.flow_from_directory(train_dir,
                                                    target_size=(300,300),
                                                    batch_size=32,
                                                    class_mode='binary')

valid_datagen = train_generator.flow_from_directory(valid_dir,
                                                    target_size=(300,300),
                                                    batch_size=32,
                                                    class_mode='binary')

## Build Model

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)))
model.add(tensorflow.keras.layers.MaxPool2D(2,2))

model.add(tensorflow.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tensorflow.keras.layers.MaxPool2D(2,2))

# model.add(tensorflow.keras.layers.Conv2D(64,(3,3),activation='relu'))
# model.add(tensorflow.keras.layers.MaxPool2D(2,2))

model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(64,activation='relu'))
model.add(tensorflow.keras.layers.Dense(1,activation='sigmoid'))
model.summary()

steps_per_epoch=1027//32 ## (number of training records/ batch_size)
validation_steps=256//32 ## (number of validation records/ batch_size)

print('steps_per_epoch: {}, validation_steps: {}'.format(steps_per_epoch,validation_steps))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(train_datagen,epochs=3,validation_data=valid_datagen,steps_per_epoch=steps_per_epoch,
          validation_steps=validation_steps)

model.evaluate(valid_datagen)


