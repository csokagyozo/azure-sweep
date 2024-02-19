import argparse
import tensorflow as tf
from tensorflow import keras


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001, help='the alpha parameter for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=15, help='the number of epochs in training')
    parser.add_argument('--architecture', default='simple_sequential')
    parser.add_argument('--dataset', default='MNIST')
    args = parser.parse_args()
    return args.epochs, args.learning_rate


def create_model(alpha=0.001):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(784,)))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dense(200, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='linear'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(alpha), metrics=['accuracy'])
    return model


epochs, alpha = get_arguments()
(X, y), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
X = X.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

model = create_model(alpha=alpha)
model.fit(X, y, epochs=epochs)
#model_untrained = keras.models.clone_model(model)
#model_untrained.save('model_untrained.h5')
#model.save('model_trained.h5')
(loss_train, accuracy_train) = model.evaluate(X, y)
(loss_test, accuracy_test) = model.evaluate(X_test, y_test)
print(loss_train)
print(accuracy_train)
print(loss_test)
print(accuracy_test)

