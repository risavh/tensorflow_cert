import tensorflow as tf

print('TF version: {}'.format(tf.version.VERSION))
print('GPU is','available' if tf.config.list_physical_devices('GPU') else 'not available')