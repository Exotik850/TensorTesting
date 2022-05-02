import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        print('Time cost:', round(end - start, 2), 's')
        return val

    return wrapper

def gen_circle_classifier_data(num):
    # generate data for circle classifier
    x_data = np.random.uniform(-1, 1, (num, 2))
    y_data = np.zeros(num)
    for i in range(num):
        if np.linalg.norm(x_data[i]) <= .5:
            y_data[i] = 1
    return x_data, y_data

def gen_model():
    # model
    layers = [
        tf.keras.layers.Dense(units=2),
        tf.keras.layers.Dense(units=6, activation='relu'), # relu
        tf.keras.layers.Dense(units=6, activation='relu'), #
        tf.keras.layers.Dense(units=1, activation='sigmoid') # sigmoid for binary classification
    ]
    model = tf.keras.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def test_model(model, x_data, y_data):
    # test model
    test_loss, test_acc = model.evaluate(x_data, y_data)
    print('Test accuracy:', test_acc)


@timer
def train_model(model, x_data, y_data, epochs):
    # train model
    model.fit(x_data, y_data, epochs=epochs, verbose=0)
    return model


if __name__ == '__main__':
    # gen data
    x_data, y_data = gen_circle_classifier_data(num_points)
    # gen model
    model = gen_model()
    # train model
    model = train_model(model, x_data, y_data, epochs=100)
    # test model
    test_model(model, x_data, y_data)
    # plot model
    num_plot = 100
    x_plot, y_plot = gen_circle_classifier_data(num_plot)
    y_plot_pred = model.predict(x_plot)
    plt.scatter(x_plot[:, 0], x_plot[:, 1], c=y_plot_pred, s=50, cmap='RdBu')

    # plot as a contour
    x_plot_contour = np.linspace(-1, 1, 100)
    y_plot_contour = np.linspace(-1, 1, 100)
    x_plot_contour, y_plot_contour = np.meshgrid(x_plot_contour, y_plot_contour)
    z_input = np.hstack([np.reshape(x_plot_contour, (-1, 1)), np.reshape(y_plot_contour, (-1, 1))])
    z_plot_pred = model.predict(z_input)
    z_plot_pred = np.reshape(z_plot_pred, (100, 100))
    plt.contour(x_plot_contour, y_plot_contour, z_plot_pred, levels=[0.5], colors='r')
    plt.show()

    # save model
    model.save('circle_classifier.h5')
    # load model

