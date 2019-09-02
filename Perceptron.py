import numpy as np
from matplotlib import pyplot as plt
import seaborn as sbs
import ActivationFunction


class Perceptron:
    """This class creates a perceptron which can be used to learn functions"""
    def __init__(self, activationfunction, args ,lr = 0.1):
        """The constructor for the perceptron receives and activation function, and a list with the desired weigths, the
        size of the input is inferred from those weights
        :parameter activationfunction: Activation function as defined in ActivationFunction class
        :parameter args: list of weights for the model
        :parameter lr: learning rate used in training
        """
        try:
            """create the weights for the model"""
            self.n = len(args)-1
            self.weights = np.random.rand(self.n+1,1)
            for (i,arg) in enumerate(args):
                """Check if any given paramter in the list is a number, if not a ValueError will be thrown"""
                arg = float(arg)
                """Replace weach weight with the ones in the args list"""
                self.weights[i,0] = arg
            """Apply activation function"""
            self.actFunc = activationfunction
            self.lr = lr
        except ValueError:
            """This exception is trown if an input is received which is not a number"""
            print("One of the Inputs is not a number")
            raise ValueError

    def feed(self, arg):
        """This method feed the perceptron with a value, if a value is not a number an exception will be raised
        :parameter arg: an array with the input to be fed to the model
        :returns: a number, the predicted result of the given input
        """

        try:
            """Create an array with a 1 in the first position to be used as the bias"""
            inputArray=[1]
            """add the input numers to the array and turn it into a numpy array"""
            for i in arg:
                i = float(i)
                inputArray.append(i)
            input = np.array(inputArray)
            """The output of the perceptron is given by y = g(Wx)
            where W is the weights vector and g the activation function
            """
            x = np.matmul(input, self.weights)
            return self.actFunc.apply(x)
        except ValueError:
            print("One of the Inputs is not a number")
            raise ValueError

    def learn(self, input, trueOutput):
        """This method modifies the weights of the model by an amount proportional to the diference between the expected
        result of the feedforwar step and the real output of the model
        :parameter input: a vector with the example to be given to the model
        :parameter trueOutput: the expected result of the given example
        """
        try:
            prediction= self.feed(input)
            diff = trueOutput - prediction
            inputArray = [1]
            for i in input:
                inputArray.append(i)
            temp = np.array(inputArray)
            temp=np.transpose(temp)
            for i in range(len(self.weights)):
                self.weights[i]+=self.lr*diff*temp[i]
        except ValueError:
            print("One of the Inputs is not a number")
            raise ValueError



def initTest():
    P = Perceptron(ActivationFunction.Sigmoid(),[1,1,1])
    inputTest= np.array([1,1,1])
    assert np.all(P.weights == inputTest)
    assert P.n == 2


if __name__ == "__main__":
    """Some test for the model"""
    #Test
    initTest()
    P = Perceptron(ActivationFunction.Sigmoid(), [1,1,1,2,1])
    before = P.feed([0,0,0,0])
    for i in range(1000):
        P.learn([0, 0, 0, 0], 0)
        current = P.feed([0,0,0,0])
        print('Result: {}, difference = {}'.format(current, before-current))
