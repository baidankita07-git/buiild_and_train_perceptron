import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_AND = np.array([0, 0, 0, 1])
y_OR = np.array([0, 1, 1, 1])

def step(x):
    return 1 if x >= 0 else 0

def train_perceptron(X, y, lr=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0.0

    for epoch in range(epochs):
        for i in range(len(X)):
            z = np.dot(X[i], weights) + bias
            prediction = step(z)
            error = y[i] - prediction
            weights += lr * error * X[i]
            bias += lr * error

    return weights, bias

def show_predictions(X, y, weights, bias, name):
    print("\\n", name)

    print("Weights:", weights)

    print("Bias:", bias)

    for i in range(len(X)):
        prediction = step(np.dot(X[i], weights) + bias)
        print(X[i], "->", prediction, "(expected", y[i], ")")

w_and, b_and = train_perceptron(X, y_AND)
w_or, b_or = train_perceptron(X, y_OR)

show_predictions(X, y_AND, w_and, b_and, "AND")
show_predictions(X, y_OR, w_or, b_or, "OR")