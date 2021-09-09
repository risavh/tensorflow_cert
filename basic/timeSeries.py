import tensorflow as tf

print('TF Version: {}'.format(tf.version.VERSION))
print('GPU is','available' if tf.config.list_physical_devices('GPU') else 'not available')

dataset=tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())


print("+"*50)
## Try-2
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5,3)

for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(),end=" ")
    print()