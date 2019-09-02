import numpy as np
import ActivationFunction
import LossFunction
import DataLoader

class NeuralNetwork:
    def __init__(self, l_h, F, C, l_a, lr =0.1):
        """"This class defines a Fully Connected FeedForward Neural network, it accept the following parameters
        F number of input neurons
        C number of output neurons
        l_h list of number of neurons in every hidden layer
        l_a list of activation functions associated with each layer, this have to implement an apply() and derivative method
        lr learning rate
        """
        self.nClasses = C
        self.nInputs = F
        """ Define a list of matrices to be used as weights in each layer, these are initialised with random values"""
        self.__in = np.random.rand(F, l_h[0])
        self.__hiddens=[]
        self.__hiddens.append(self.__in)
        for i in range(len(l_h)-1):
            self.__hiddens.append(np.random.rand(l_h[i], l_h[i + 1]))
        self.__hiddens.append(np.random.rand(l_h[-1], C))
        """Define a Vector of Activaction functions"""
        self.__activationFunctions = [f for f in l_a]
        self.__out = np.random.rand(C, 1)
        """Define the vector wich will contain the bias values, these are also initialised with random values"""
        self.__b = []
        for i in range(len(l_h)):
            self.__b.append(np.random.rand(1, l_h[i]))
        self.__c = np.random.rand(1, C)
        self.__b.append(self.__c)
        """define list where activations and outputs of each layer will be stored after a feedforward step, these are used to compute the gradients in training"""
        self.__activations = []
        self.__outputs = []
        self.lr = lr
        self.lastLoss= None


    def feed(self, x):
        """
        This method is used to perform a forward pass to the network
        :param x: A list-like object represent te input to the network
        :return: A numpy array, the output of the network
        """
        """Clean the activations and outputs storde int he class"""
        self.__activations=[]
        self.__outputs = []
        """add the input to the outputs array"""
        self.__outputs.append(x.reshape(1,x.shape[0]))
        h = self.__outputs[0]
        for (layer, activation, b) in zip(self.__hiddens, self.__activationFunctions,self.__b):
            """
            In this loop the forward propagation is done for each layer using
            y = f(Wh+b)
            """
            h = np.matmul(h, layer)
            h = np.add(h,b)
            self.__activations.append(h)
            h=activation.apply(h)
            self.__outputs.append(h)
        return h

    def train(self, input, expected):
        """
        This method modifies the weights of the network in the direction of fastest descent of a loss funciton, usint the backpropagation algorithm
        :param input: a list-like object wich contains the features of a trainnig example
        :param expected: a list-like object wich contains the labels of the training example
        """
        expected = np.array(expected).reshape(1,self.nClasses)
        """First the example is feed to the network and the prediction is stored"""
        result = self.feed(input)
        """Compute the current loss using Mean Squared Error"""
        lossFunction = LossFunction.MeanSquareError()
        mse = lossFunction.apply(expected, result)
        self.lastLoss=mse
        """Compute the gradient of the loss function"""
        msegradient = lossFunction.derivative(expected,result)

        """Create a list to store de deltas of each layer"""
        deltas = [None] * (len(self.__hiddens))
        deltaBias = [None] *(len(self.__b))
        """Copute the deltas for the output layer"""
        deltas[-1] = np.matmul(self.__outputs[-1],np.matmul(msegradient,self.__activationFunctions[-1].derivative(self.__activations[-1])))
        deltaBias[-1] = np.matmul(self.__outputs[-1],np.matmul(msegradient,self.__activationFunctions[-1].derivative(self.__activations[-1])))

        for i in reversed(range(len(deltaBias)-1)):
            """
            Compute the deltas for the other layers starting by the second to last one, the deltas are given by:
            d = dL/dg*dg/da*da/dw
            where:
            dl/dg = Loss funciton derivative with respect to last layer outputs
            dg/da = Activation Function derivative with respect to activations
            da/dw = activation derivative with respect to the parameters
            
            The deltas are split into the deltas of the weiths and the delta of the bias term of each neuron
            """
            deltas[i] = np.matmul(
                np.matmul(self.__activationFunctions[i].derivative(self.__activations[i]).T,self.__outputs[i+1]).T,
                np.matmul(np.array(self.__hiddens[i+1]), np.array(deltas[i+1]).T)).T
            deltaBias[i] = np.matmul(
                np.matmul(self.__activationFunctions[i].derivative(self.__activations[i]).T, np.ones(deltaBias[i+1].shape)),
                np.sum(np.matmul(np.array(self.__b[i+1]).T, np.array(deltaBias[i + 1])), axis= 1).reshape(deltaBias[i+1].shape).T).T

        for j in reversed(range(len(self.__hiddens))):
            """ Update the error terms by adding to each weight the deltas, scaled by the learning rate of the network"""
            self.__hiddens[j] += self.lr * np.matmul(deltas[j].T,self.__outputs[j]).T
            self.__b[j] += self.lr * deltaBias[j]

    def DataSetLoss(self, lossFunction, features, labels):
        """This method computes the loss of the network with respect to a particular dataset, it computes the loss of ecery example
        and outputs the mean
        :param lossFunction: loss Function to be used
        :param features: features of dataset
        :param labels: labels of the dataset
        """
        acc = 0
        for i in range(len(features)):
            """for each example, predict the class and compute loss"""
            predicted = self.feed(features[i])
            acc += lossFunction.apply(np.array(labels[i]).reshape(predicted.shape), predicted)
        """output the mean of every loss"""
        return acc/len(features)

    def GenerateConfusionMatrix(self, features, labels):
        "This method allows to generate a confussion matrix for "
        c = self.nClasses
        matrix = np.zeros([c, c])
        for i in range(len(features)):
            predicted = np.argmax(self.feed(features[i]))
            print(self.feed(features[i]))
            label = np.argmax(labels[i])
            matrix[predicted,label] += 1
        return matrix

if __name__== "__main__":
    """Load the dataset"""
    data = DataLoader.Dataset("Data/iris.csv")
    """normalize and onehot encode the dataset"""
    data.normalization()
    data.OneHot()
    """Split the dataset in train and test subsets"""
    data.trainTestSplit(0.33)
    """initialize activation functions"""
    sig = ActivationFunction.Sigmoid()
    tanh = ActivationFunction.Tanh()

    print("Number of features: {}, number of classes: {}".format(data.nFeatures(), data.nClasses))
    """Initialize the network"""
    nn = NeuralNetwork([10, 10, 10], data.nFeatures() , data.nClasses, [sig, sig, sig, sig], lr = 0.1)
    """Initialize lost function"""
    lossFunction = LossFunction.MeanSquareError()
    """Train the network on 1000 epochs over the training data"""
    for j in range(1000):
        """feed each ecample to the dataset"""
        for i in range(len(data.trainFeatures)):
            nn.train(data.trainFeatures[i], data.trainLabels[i])
        """Compute losses"""
        trainLoss = nn.DataSetLoss(lossFunction, data.trainFeatures, data.trainLabels)
        testLoss = nn.DataSetLoss(lossFunction, data.testFeatures, data.testLabels)
        print("epoch: {}, train loss: {}, validation loss: {}".format(j, trainLoss, testLoss))
    """Show confussion matrix"""
    print(nn.GenerateConfusionMatrix(data.testFeatures, data.testLabels))