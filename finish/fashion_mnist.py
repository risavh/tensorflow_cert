import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.python.keras.models
from tensorflow.keras.datasets import fashion_mnist

print('TF Version: {}'.format(tf.version.VERSION))

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print('X Train Shape: {} y Train Shape: {}'.format(x_train.shape, y_train.shape))
print('X Test Shape: {} y Test Shape: {}'.format(x_test.shape, y_test.shape))

prod_dict = {0: 'T-shirt/top',
             1: 'Trouser',
             2: 'Pullover',
             3: 'Dress',
             4: 'Coat',
             5: 'Sandal',
             6: 'Shirt',
             7: 'Sneaker',
             8: 'Bag',
             9: 'Ankle boot'}

mapping_dict = {k: (list((y_train == k).nonzero())[0][0]) for k in np.unique(y_train[:25])}

print(mapping_dict)

# fig, ax = plt.subplots(2, 5, figsize=(15, 7))
# cnt = 0
# for i in range(2):
#     for j in range(5):
#         ax[i, j].imshow(x_train[mapping_dict[cnt]])
#         ax[i, j].set_title(prod_dict[cnt])
#         ax[i, j].grid(False)
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
#         cnt += 1
#
# plt.axis("off")
# plt.tight_layout()
# plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train= x_train.reshape(60000, 28, 28,1)
x_test= x_test.reshape(10000, 28, 28,1)

## Callback

class myCallbacks(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >0.87:
            print('This is awesome, kill it \n')
            self.model.stop_training=True

callbacks=myCallbacks()


model = tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Conv2D(16,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(tensorflow.keras.layers.MaxPool2D((2,2)))

model.add(tensorflow.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tensorflow.keras.layers.MaxPool2D((2,2)))

model.add(tensorflow.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tensorflow.keras.layers.MaxPool2D((2,2)))

model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,callbacks=[callbacks])

model.evaluate(x_test,y_test)



