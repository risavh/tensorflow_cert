import tensorflow as tf

print('TF Version: {}'.format(tf.version.VERSION))
print('GPU is', 'available' if tf.config.list_physical_devices('GPU') else 'not available')

dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())

print("+" * 50)
## Try-2
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)

for window_dataset in dataset:
    # print("t===",window_dataset)
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()

print("+3" * 50)
## Try-3

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)
dataset = dataset.flat_map(lambda x: x.batch(5))

for window in dataset:
    print(window.numpy())

print("+4" * 50)
## Try-4

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, 1, drop_remainder=True)
dataset = dataset.flat_map(lambda j: j.batch(5))
dataset = dataset.map(lambda g: (g[:-1], g[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print(x.numpy(), y.numpy())

print("*window" * 50)
print("\n\n")
## Windowed Fxn

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda x: x.batch(5))
dataset = dataset.shuffle(buffer_size=10).map(lambda x: (x[:-1], x[-1:]))
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print(x.numpy(), y.numpy(), end='')
    print()


def window_fxn(series, window_size, batch_size, buffer_size):
    data = tf.data.Dataset.from_tensor_slices(series)
    data = data.window(window_size + 1, shift=1, drop_remainder=True)
    data = data.shuffle(buffer_size=buffer_size).map(lambda k: (k[:-1], k[-1:]))
    data = data.batch(batch_size).prefetch(1)
    return data
