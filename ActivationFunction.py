import numpy as np

class Sigmoid:
    """This class defines the Sigmoid function"""
    def __init__(self):
        pass

    def apply(self, x):
        """This method applies the function to a certain input
        :parameter: x, list-like numeric inputs"""
        x = np.array(x)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """This method applies the derivative of the function to a certain input
        :parameters: x list-like numeric inputs"""
        x = np.array(x)
        return self.apply(x) * (1 - self.apply(x))


class Step:
    """This class defines the Step function"""
    def __init__(self):
        pass

    def apply(self, x):
        """This method applies the function to a certain input
        :parameter: x, list-like numeric inputs"""
        if x >= 0:
            return 1
        else:
            return 0

    def derivative(self, x):
        """This method applies the derivative of the function to a certain input
        :parameters: x list-like numeric inputs"""
        if np.any( x== 0):
            raise ZeroDivisionError
        else:
            return 0


class Tanh:
    """This class defines the Tanh function"""
    def __init__(self):
        pass

    def apply(self, x):
        """This method applies the function to a certain input
        :parameter: x, list-like numeric inputs"""
        x = np.array(x)
        return (np.exp(x) - np.exp(-x) ) / (np.exp(x) + np.exp(-x))

    def derivative(self, x):
        """This method applies the derivative of the function to a certain input
        :parameters: x list-like numeric inputs"""
        x = np.array(x)
        return 1 - np.power(self.apply(x), 2)

class SoftMax():
    """This class defines the SoftMax function"""
    def __init__(self):
        pass

    def apply(self,x):
        """This method applies the function to a certain input
        :parameter: x, list-like numeric inputs"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def derivative(self, s):
        """This method applies the derivative of the function to a certain input
        :parameters: x list-like numeric inputs"""
        jacobian_m = np.diag(s)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1 - s[i])
                else:
                    jacobian_m[i][j] = -s[i] * s[j]
        return jacobian_m

if __name__=="__main__":

    """"Some test of the functions"""
    sig = Sigmoid()
    print(sig.apply([1,1,1]))
    print(sig.derivative([1,1,1]))
    tanh = Tanh()
    print(tanh.apply([1,1,1]))
    print(tanh.derivative([1,1,1]))
    soft = SoftMax()
    print(soft.apply(np.array([1, 2])))
    print(soft.derivative(np.array([1, 2])))