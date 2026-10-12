import numpy as np

# XOR dataset: inputs and expected outputs
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])
# Start with random weights and bias 
weights = np.random.rand(2)
bias = np.random.rand(1)
lr = 0.1
# Train for 20 epochs
for epoch in range (20):
    for i in range(len(X)):
        z=np.dot(X[i],weights)+bias
        prediction = 1 if z >= 0 else 0
        error = y[i] - prediction
        weights += lr * error * X[i]
        bias += lr * error

# Test the trained perceptron
print("Final weights:", weights)
print("Final bias:", bias)
print("\nPredictions:")
for i in range(len(X)):
    z = np.dot (X[i], weights) + bias
    prediction = 1 if z >= 0 else 0
    print(X[i], "->", prediction, "(expected", y[i], ")")