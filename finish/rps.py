import os
import zipfile
import tensorflow as tf
import tensorflow.keras.preprocessing.image

zip_training = r"D:\works\koglu\datasets\rps\rps.zip"
zip_valid = r"D:\works\koglu\datasets\rps\rps-test-set.zip"

train_dir = r"D:\works\koglu\datasets\rps\training"
valid_dir = r"D:\works\koglu\datasets\rps\validation"
zipref = zipfile.ZipFile(zip_training, "r")
zipref.extractall(train_dir)
zipref = zipfile.ZipFile(zip_valid, "r")
zipref.extractall(valid_dir)

train_path = r"D:\works\koglu\datasets\rps\training\rps"
valid_path = r"D:\works\koglu\datasets\rps\validation\rps-test-set"

import random
import os
import matplotlib.image as mpimg


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
labels = ['paper', 'rock', 'scissors']
dir = train_path
title = 'Training Data'
# visualize_data(labels,dir,title)

train_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.255,
                                                                          rotation_range=20,
                                                                          width_shift_range=0.2,
                                                                          height_shift_range=0.2,
                                                                          shear_range=0.2,
                                                                          zoom_range=0.2,
                                                                          horizontal_flip=True,
                                                                          fill_mode='nearest')

valid_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.255)

train_datagen = train_generator.flow_from_directory(train_path,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='categorical')

valid_datagen = valid_generator.flow_from_directory(valid_path,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='categorical')

model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(tensorflow.keras.layers.MaxPool2D(2, 2))

model.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tensorflow.keras.layers.MaxPool2D(2, 2))

# model.add(tensorflow.keras.layers.Conv2D(64,(3,3),activation='relu'))
# model.add(tensorflow.keras.layers.MaxPool2D(2,2))

model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(64, activation='relu'))
model.add(tensorflow.keras.layers.Dense(3, activation='softmax'))
model.summary()

# steps_per_epoch=1027//32 ## (number of training records/ batch_size)
# validation_steps=256//32 ## (number of validation records/ batch_size)
steps_per_epochs = 2520 // 32
validation_steps = 372 // 32
print('steps_per_epoch: {}, validation_steps: {}'.format(steps_per_epochs, validation_steps))

# AUTOTUNE = tf.data.AUTOTUNE
#
# train_datagen = train_datagen.prefetch(buffer_size=AUTOTUNE)
# valid_datagen = valid_datagen.prefetch(buffer_size=AUTOTUNE)
# #test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

model.compile(loss=tensorflow.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

history = model.fit(train_datagen, epochs=40, validation_data=valid_datagen, steps_per_epoch=steps_per_epochs,
                    validation_steps=validation_steps)

model.evaluate(valid_datagen)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
epochs = range(len(history.history['accuracy']))

ax[0].plot(epochs, history.history['accuracy'], label='train', lw=2, marker='o', color='tomato')
ax[0].plot(epochs, history.history['val_accuracy'], label='val', lw=2, marker='*', color='teal')
ax[0].set_title('Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()

ax[1].plot(epochs, history.history['loss'], label='train', lw=2, marker='o', color='tomato')
ax[1].plot(epochs, history.history['val_loss'], label='val', lw=2, marker='o', color='teal')
ax[1].set_title('Loss')
ax[1].legend()

ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
plt.suptitle('Learning Curves')
plt.show()
