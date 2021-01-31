import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, up, down):
        self.up = up
        self.down = down

        def feedforward(self, inputs):
            total = np.dot(self.up, inputs) + self.bias
            return sigmoid(total)


up = np.array([0, 1])
bias = 4
n = Neuron(up, bias)

x = np.array([2, 3])
print(n.feedforward)


class Neuron2:
    def __init__(self, fast, stop):
        self.fast = fast
        self.stop = stop

        def feedforward(self, inputs):
            total = np.dot(self.fast, inputs) + self.bias
            return sigmoid(total)


fast = np.array([0, 1])
bias = 5
n = Neuron2(fast, bias)

x = np.array([5, 8])
print(n.feedforward)
