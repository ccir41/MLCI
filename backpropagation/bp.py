import numpy as np

input_sequence = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_sequence = np.array([[0], [1], [1], [0]])


class NeuralNetwork():
    def __init__(self):
        self.inputNeurons = 2
        self.hiddenNeurons = 2
        self.outputNeurons = 1
        self.learning_rate = 0.1

        self.input_to_hidden_weight = np.random.uniform(
            size=(self.inputNeurons, self.hiddenNeurons))
        self.input_to_hidden_bias = np.random.uniform(
            size=(1, self.hiddenNeurons))
        self.hidden_to_output_weight = np.random.uniform(
            size=(self.hiddenNeurons, self.outputNeurons))
        self.hidden_to_output_bias = np.random.uniform(
            size=(1, self.outputNeurons))

    def forward(self, input_sequence):
        self.hidden_layer_activation = np.dot(
            input_sequence, self.input_to_hidden_weight)
        self.hidden_layer_activation += self.input_to_hidden_bias
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(
            self.hidden_layer_output, self.hidden_to_output_weight)
        self.output_layer_activation += self.hidden_to_output_bias
        predicted_output = self.sigmoid(self.output_layer_activation)
        return predicted_output

    def sigmoid(self, activation):
        return 1.0/(1.0 + np.exp(-activation))

    def sigmoidPrime(self, output):
        return output*(1.0 - output)

    def backward(self, input_sequence, output_sequence, predicted_output):
        self.error = output_sequence - predicted_output
        self.output_delta = self.error*self.sigmoidPrime(predicted_output)

        self.hidden_layer_output_error = self.output_delta.dot(
            self.hidden_to_output_weight.T)
        self.hidden_delta = self.hidden_layer_output_error * \
            self.sigmoidPrime(self.hidden_layer_output)

        self.input_to_hidden_weight += input_sequence.T.dot(
            self.hidden_delta) * self.learning_rate
        self.input_to_hidden_bias += np.sum(self.hidden_delta,
                                            axis=0, keepdims=True) * self.learning_rate
        self.hidden_to_output_weight += self.hidden_layer_output.T.dot(
            self.output_delta)*self.learning_rate
        self.hidden_to_output_bias += np.sum(
            self.output_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, input_sequence, output_sequence):
        predicted_output = self.forward(input_sequence)
        self.backward(input_sequence, output_sequence, predicted_output)
        return predicted_output

    def parametersValue(self):
        print("Input to hidden weights: \n", self.input_to_hidden_weight)
        print("Input to hidden bias: \n", self.input_to_hidden_bias)
        print("Hidden to output weights: \n", self.hidden_to_output_weight)
        print("Hidden to output bias: \n", self.hidden_to_output_bias)
        print(
            "Loss: " + str(np.mean(np.square(output_sequence - nn.forward(input_sequence)))))


nn = NeuralNetwork()

print("####################################")
print("Before Training: ")
print("####################################")
nn.parametersValue()

for i in range(20000):
    predicted_output = nn.train(input_sequence, output_sequence)
    print("Predicted Output")
    print(predicted_output)

print("\n####################################")
print("After Training: ")
print("####################################")
nn.parametersValue()
