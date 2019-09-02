import numpy as np
from sklearn.model_selection import train_test_split

class Dataset:
    """This Class defines a dataset and some key utilities for them"""
    def __init__(self,path, oneHot = False, norm = False ):
        """The constructor for this class receibes the path to a .csv file with the data
        :parameter path: a string containing the path of the data file
        :parameter oneHot: False on default, set to True if the dataset labels are in One-hot encoding
        :parameter norm: False on default, set to True if the dataset is already normalized"""
        self.data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1)
        self.labels = self.data[:,-1]
        self.features = self.data[:,0:-1]
        self.onehot = oneHot
        self.norm = norm
        self.nClasses = int(max(self.labels)+1)
        """These matrix are used for splitting the dataset in train and test"""
        self.trainFeatures = None
        self.trainLabels = None
        self.testFeatures = None
        self.testLabels = None

    def __len__(self):
        """The lenght of the dataset is the ammount of examples contained in it"""
        return self.data.shape[0]

    def nFeatures(self):
        """:returns: number of features of the dataset"""
        if self.onehot:
            return int(self.data.shape[1]-self.nClasses)
        else:
            return self.data.shape[1]-1

    def print(self):
        """This method prints the data as a matrix"""
        print(self.data)

    def normalization(self):
        """This method is used to normalize the features of the dataset,
        if the dataset is already normalized, this method does nothing"""
        if not self.norm:
            """Creates an auxiliary array for doing operations"""
            temp=self.data
            for i in range(temp.shape[1]-1):
                """Apply min-max normalization on each column of the dataset, excluding label columns"""
                column = self.data[:,i]
                max = column.max()
                min = column.min()
                normColumn = (column - min)/(max-min)
                temp[:, i] = normColumn
            self.data = temp
            """Set the norm parameter to true"""
            self.norm = True

    def OneHot(self):
        """Applies on-hot encoding to the labels of the dataset,
        if the dataset is already in onehot encoded the method does nothing """
        if not self.onehot:
            """extract labels from data, and useful information as number of classes"""
            labels=self.data[:,-1]
            n = int(self.data[:,-1].max())
            """create the new label matrix with the same number of rows as number of examples, and as many columns
            as labels, initialize this matrix with 0"""
            a, b = self.data.shape
            temp = np.zeros((a, b + n))

            temp[:,:-n]= self.data

            for (example,i) in zip(labels,range(len(labels))):
                """For each example in the dataset put a 1 on the labels matrix based on the dataset label"""
                temp[i, b-1] = 0
                temp[i, b+int(example)-1] = 1
            """replace the old labels with the new label matrix"""
            self.data = temp
            self.labels = []
            for i in range(n+1):
                self.labels.append((self.data[:,b+i-1]).tolist())
            self.labels = np.array(self.labels).T
            """Set the the onehot parameter to True"""
            self.onehot = True

    def trainTestSplit(self, testSize):
        """This method allows to split the dataset in test and train suing sklearn library"""
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size = testSize)
        self.trainFeatures = X_train
        self.trainLabels = y_train
        self.testFeatures = X_test
        self.testLabels = y_test



if __name__=='__main__':
    """Some test of the functions"""
    Data = Dataset("Data/pulsar_stars.csv")
    Data.normalization()
    print(Data.nFeatures())
    Data.OneHot()
    print(Data.labels)

