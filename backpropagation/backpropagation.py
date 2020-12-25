""" Implementation of backpropagation algorithm from scratch """

# importing libraries
from numpy.random import RandomState
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


np.random.seed(10)


class NeuralNetwork:
    """ Neural Network parameters """

    def __init__(self, _in=2, _hl=[2, 2], _on=1, _lr=0.1):
        """
        Initilizations

        args:
            _in : number of input neurons
            _hl : list containing hidden neurons at each hidden layers e.g. [3, 3]
            _on : number of output neurons
            _lr : learning rate
        """

        self.input_neurons = _in
        self.hidden_layers = _hl
        self.output_neurons = _on
        self.learning_rate = _lr
        self.weights = []
        self.biases = []
        self.activations = []
        self.derivatives = []
        self.delta = []

        # general representation of layers [2, 2, 2, 1]
        layers = [_in] + _hl + [_on]

        # initialize weights and biases
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i+1])
            self.weights.append(weight)
            # Biases not for input layers
            bias = np.random.rand(1, layers[i+1])
            self.biases.append(bias)
        # print(self.weights)
        # print(self.biases)

        # activations
        for i in range(len(layers)):
            activation = np.zeros(layers[i])
            self.activations.append(activation)
        # print(self.activations)

        # activations derivative
        for i in range(len(layers) - 1):
            derivative = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(derivative)
        # print(self.derivatives)

        # delta
        for i in range(len(layers) - 1):
            delta = np.zeros((layers[i], layers[i+1]))
            self.delta.append(delta)
        # print(self.delta)

    def sigmoid(self, x):
        """ Sigmoid activation function """

        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivatives(self, x):
        """ Sigmoid derivative """

        return x * (1.0 - x)

    def mse(self, error):
        """ Mean square loss """

        return np.average(error**2)

    def gradient_descent(self):
        """ Update weights and biases """

        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i]
            self.biases[i] += self.delta[i]

    def forward(self, inputs):
        """ Forward propagate inputs and produce predicted outputs """

        # activations for first layer is inputs itself
        activations = inputs
        # cache activations for backpropagation
        self.activations[0] = activations

        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            bias_reshaped = bias.reshape(-1)  # [[0.59 0.62]] -> [0.59 0.62]
            net_inputs = np.dot(activations, weight) + bias_reshaped
            activations = self.sigmoid(net_inputs)
            # cache activations for backpropagation
            self.activations[i+1] = activations
        return activations

    def backward(self, error):
        """ Backpropagate error """

        # iterate through backward
        for i in reversed(range(len(self.derivatives))):
            activation = self.activations[i+1]
            delta = error * self.sigmoid_derivatives(activation)
            self.delta[i] = delta * self.learning_rate
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations = current_activations.reshape(
                current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(
                current_activations, delta_reshaped) * self.learning_rate
            error = np.dot(delta, self.weights[i].T)

    def train(self, inputs, targets, epochs):
        """ Model training """
        errors = []
        for i in range(epochs):
            sum_errors = 0
            for j, _input in enumerate(inputs):
                target = targets[j]
                predicted_output = self.forward(_input)
                error = target - predicted_output
                sum_errors += self.mse(error)
                self.backward(error)
                # update weights and biases
                self.gradient_descent()
            print(f"Error: {sum_errors/len(X)} at epoch {i+1}")
            errors.append(sum_errors)
        print("\nTraining complete...")
        plt.plot(errors)
        plt.xlabel("No of Epochs")
        plt.ylabel("Error values")
        plt.title("Plot of Mean square error versus number of Epoch")
        plt.show()

    def test(self):
        """ Test the model accuracy """
        pass

    def make_datasets(self, filename):
        """ Import datasets and split into training and testing """

        df = pd.read_csv(filename)
        train = df.sample(frac=0.9, random_state=RandomState())
        test = df.loc[~df.index.isin(train.index)]
        train.to_csv('train.csv', index=False)
        test.to_csv('test.csv', index=False)

    def import_datasets(self, train=True):
        """ Import datasets for training and testing """
        if train:
            df = pd.read_csv('./train.csv')
        else:
            df = pd.read_csv('./test.csv')
        X, y = df.loc[:, df.columns != 'class'].to_numpy(
            dtype=float), df['class'].to_numpy(dtype=int)
        # X = (X-X.min())/(X.max()-X.min())
        # mean normailization
        X = (X-X.mean())/X.std()
        distinct_y = list(set(y))
        # if multi class problem then encode output
        if len(distinct_y) > 2:
            encoded_output = self.encode_output(y, distinct_y)
            return X, encoded_output
        y = y.reshape(y.shape[0], -1)
        return X, y

    def encode_output(self, outputs, distinct_y):
        """ Encode multiclass output """

        labels = []
        for output in outputs:
            encoded = np.zeros((len(distinct_y)))
            for i in range(len(distinct_y)):
                if i+1 == output:
                    encoded[i] = 1
            labels.append(encoded)
        return np.asarray(labels)  # convert python list to numpy array

    def decode_output(self, encoded_output, test=True, threshold=0.5):
        """ Decode Encoded output """

        if not test:
            encoded_output = np.where(encoded_output > threshold, 1, 0)
        return [i+1 for eo in encoded_output for i, o in enumerate(eo) if o == 1]

    def model_accuracy(self, actual, predicted):
        """ Calculate model accuracy """

        count = 0
        for act, pred in zip(actual, predicted):
            if act == pred:
                count += 1
        return count / len(actual)


if __name__ == "__main__":
    """ Run Script """

    """
        7 features so 7 input neurons
        hidden layers of our choice 
        3 class of wheat so 3 output neurons
        learning rate of our choice  
    """
    nn = NeuralNetwork(7, [5, 5], 3, 0.3)
    filename = './seeds.csv'
    nn.make_datasets(filename)
    X, y = nn.import_datasets()
    nn.train(X, y, 3000)
    test, target = nn.import_datasets(train=False)
    output = nn.forward(test)
    decoded_output = nn.decode_output(output, test=False, threshold=0.75)
    target = nn.decode_output(target, threshold=0.75)
    accuracy = nn.model_accuracy(target, decoded_output)
    print(f"\nAccuracy of the model is {accuracy*100}%")

    """
    # XOR gate
    X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(2, [2], 1, 0.1)
    nn.train(X, y, 50000)
    output = nn.forward(X)
    decoded_output = nn.decode_output(output, test=False, threshold=0.8)
    target = nn.decode_output(y, threshold=0.8)
    accuracy = nn.model_accuracy(target, decoded_output)
    print(f"\nAccuracy of the model is {accuracy*100}%")
    """
