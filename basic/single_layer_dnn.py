import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('TF version: {}'.format(tf.version.VERSION))
print('GPU is', 'available' if tf.config.list_physical_devices('GPU') else 'not available')


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

print(type(series))
print(series)


def window_fxn(series, window_size, batch_size, shuffle_buffer_size):
    data = tf.data.Dataset.from_tensor_slices(series)
    data = data.window(window_size + 1, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(window_size + 1))
    data = data.shuffle(buffer_size=shuffle_buffer_size).map(lambda k: (k[:-1], k[-1]))
    data = data.batch(batch_size).prefetch(1)
    return data


dataset = window_fxn(x_train, window_size, batch_size, shuffle_buffer_size)

# cnt = 0
# for x, y in dataset:
#     print(x.numpy(), y.numpy(), end='')
#     cnt += 1
#     if cnt > 2:
#         break

lo = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([lo])

model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))

model.fit(dataset, epochs=100, verbose=0)

print("Layer Weight : {}".format(lo.get_weights()))

print(series[1:21].shape)
print(series[1:21][np.newaxis].shape)

print('Prediction: {}'.format(model.predict(series[1:21][np.newaxis])))

forecast = []

for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

##print(forecast.shape)

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(time_valid, x_valid)
ax.plot(time_valid, results)
plt.title('Model-0: 1 Dense Layers')

plt.show()

print('MAE Model-0: 1 Dense Layers= {}'.format(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()))

## Change Model add new Dense Layers

model1 = tf.keras.models.Sequential()
model1.add(tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'))
model1.add(tf.keras.layers.Dense(10, activation='relu'))
model1.add(tf.keras.layers.Dense(1))

model1.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9))

history = model1.fit(dataset, epochs=100, verbose=0)

forecast = []

for time in range(len(series) - window_size):
    forecast.append(model1.predict(series[time:time + window_size][np.newaxis]))

##print(forecast.shape)

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(time_valid, x_valid)
ax.plot(time_valid, results)
plt.title('Model-1: 3 Dense Layers')
plt.show()

print('MAE Model-1: 3 Dense Layers = {}'.format(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()))

## Model-2 with lr schedule

model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'))
model2.add(tf.keras.layers.Dense(10, activation='relu'))
model2.add(tf.keras.layers.Dense(1))

lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))

model2.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=0.9))

history = model2.fit(dataset, epochs=100, callbacks=[lr_schedular], verbose=0)

forecast = []

for time in range(len(series) - window_size):
    forecast.append(model2.predict(series[time:time + window_size][np.newaxis]))

##print(forecast.shape)

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(time_valid, x_valid)
ax.plot(time_valid, results)
plt.title('Model-2: 3 Dense Layers with lr')
plt.show()

lr_vals = 1e-8 * 10 ** (np.arange(100) / 20)
plt.semilogx(lr_vals, history.history['loss'])
plt.axis([1e-8, 1e-3, 0, 300])
plt.show()

print('MAE Model-2: 3 Dense Layers with Lr Schedular= {}'.format(
    tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()))

## Model-3 with lr schedule

model3 = tf.keras.models.Sequential()
model3.add(tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'))
model3.add(tf.keras.layers.Dense(10, activation='relu'))
model3.add(tf.keras.layers.Dense(1))

# lr_schedular = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/20))

model3.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=1.6e-5, momentum=0.9))

history = model3.fit(dataset, epochs=500, verbose=0)

forecast = []

for time in range(len(series) - window_size):
    forecast.append(model3.predict(series[time:time + window_size][np.newaxis]))

##print(forecast.shape)

forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(time_valid, x_valid)
ax.plot(time_valid, results)
plt.title('Model-2: 3 Dense Layers with lr')
plt.show()

print('MAE Model-3: 3 Dense Layers After Fixing LR= {}'.format(
    tf.keras.metrics.mean_absolute_error(x_valid, results).numpy()))
