import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


datadict_bach_1 = unpickle('data_batch_1')
datadict_bach_2 = unpickle('data_batch_2')
datadict_bach_3 = unpickle('data_batch_3')
datadict_bach_4 = unpickle('data_batch_4')
datadict_bach_5 = unpickle('data_batch_5')
datadict_test = unpickle('test_batch')

bach_1_data, bach_1_labels = datadict_bach_1["data"], datadict_bach_1["labels"]
bach_2_data, bach_2_labels = datadict_bach_2["data"], datadict_bach_2["labels"]
bach_3_data, bach_3_labels = datadict_bach_3["data"], datadict_bach_3["labels"]
bach_4_data, bach_4_labels = datadict_bach_4["data"], datadict_bach_4["labels"]
bach_5_data, bach_5_labels = datadict_bach_5["data"], datadict_bach_5["labels"]

train_all_data = np.concatenate(
    (bach_1_data, bach_2_data, bach_3_data, bach_4_data, bach_5_data))
train_labels_all_data = np.concatenate((bach_1_labels, bach_2_labels,
                                        bach_3_labels, bach_4_labels,
                                        bach_5_labels))

test_data = datadict_test["data"]
test_labels_raw = datadict_test["labels"]

train_images, test_images = train_all_data / 255.0, test_data / 255.0
train_labels, test_labels = keras.utils.to_categorical(train_labels_all_data,
                                                       num_classes=10), keras.utils.to_categorical(
    test_labels_raw, num_classes=10)

mlp = keras.models.Sequential([


    # Hidden layer with 250 neurons
    keras.layers.Dense(250, input_dim=3072, activation='sigmoid'),

    # Last layer with 10 full-connected neurons
    keras.layers.Dense(10, input_dim=3072, activation='sigmoid')
])

# Compile the model
keras.optimizers.SGD(learning_rate=0.3)
mlp.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)
number_of_epochs = 200
# Training of the model
tr_hist = mlp.fit(train_images, train_labels, epochs=number_of_epochs,
                  verbose=1)


# With 5 hidden layers, learning rate 0.3 and 10 epochs
# accuracy is 31.45% for training data
# and 31.54% for test data.
# 20 Hidden layers, learning rate of 0.4 and 10 epochs
# training accuracy: 39.53% test accuracy:39.46%
# 150 hidden layers, learning rate 0.2 and 20 epochs
# trainign accuracy: 45.38% test accuracy: 44.47%

def predict(data_images, data_labels):
    predicted_labels = np.argmax(mlp.predict(data_images), axis=1)
    true_labels = np.argmax(data_labels, axis=1)
    return accuracy_score(true_labels, predicted_labels)


# Calculate training accuracy
train_accuracy = predict(train_images, train_labels)
print(f'Classication accuracy (training data): {train_accuracy * 100:.2f}%')

# Calculate testing accuracy
test_accuracy = predict(test_images, test_labels)
print(f'Classication accuracy (test data): {test_accuracy * 100:.2f}%')

plt.plot(tr_hist.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training'], loc='upper right')
plt.show()
