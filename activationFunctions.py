import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

step = np.where(x >= 0, 1, 0)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

plt.plot(x, step, label="Step")
plt.plot(x, sigmoid, label="Sigmoid")
plt.plot(x, tanh, label="Tanh")
plt.plot(x, relu, label="ReLU")

plt.title("Activation Functions")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Activation Functions")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("activation_functions_short.png")
