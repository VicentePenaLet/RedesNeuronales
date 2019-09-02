import numpy as np



class MeanSquareError:
    """Mean squared error function"""
    def __init__(self):
        pass

    def apply(self, expected, true):
        """This method computes the Mean squared error between 2 vectors, this vectors have to have the same dimensions
        :parameter expected: first vector, the true labels of the model
        :parameter true: second vector, the predicted values of the model
        :returns: a number, the mean squared error between the input vectors"""
        if len(expected) != len(true):
            """Check if vectors have the same size, exits if not"""
            print("Vector lenghts dont match")
            exit(1)
        else:
            """Computation of the mean squared error by definition"""
            acc = 0
            for y_hat,y in zip(expected, true):
                acc += np.sum(np.power(y-y_hat,2))
            return acc/np.max(np.array(true).shape)

    def derivative(self, expected, true):
        """This method computed the derivative of the mean squared error function
         :parameter expected: first vector, the true labels of the model
        :parameter true: second vector, the predicted values of the model
        :returns: a matrix, the mean squared error derivative between the input vectors"""
        if len(expected) != len(true):
            """Check if vectors have the same size, exits if not"""
            print("Vector lenghts dont match")
            exit(1)
        else:
            """Return the derivative of the function"""
            return 1/2*(expected-true).T

def applyTest():
    mse = MeanSquareError()
    true = [0, 0, 0, 0, 0]
    assert mse.apply(true, true) == 0
    assert mse.apply(true, [1,1,1,1,1]) == 1

if __name__ == "__main__":
    """Some test, this should output:
        Vector lenghts dont match
        Process finished with exit code 1"""
    applyTest()
    mse = MeanSquareError()
    true = [0, 0, 0, 0, 0]
    mse.apply(true, [1,1])

