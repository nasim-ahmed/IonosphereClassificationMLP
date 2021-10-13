from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

class CSVDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        df = read_csv(path, header=None)
        self.X = df.values[:,:-1]
        self.Y = df.values[:, -1]
        self.X = self.X.astype('float32')
        #label encode the target variable
        self.Y = LabelEncoder.fit_transform(self.Y)
        self.Y = self.Y.astype('float32')
        self.Y = self.Y.reshape(len(self.Y), 1)

    # get a length of the dataset
    def __len__(self):
        return len(self.X)
    
    #get a row at an index
    def __getitem__(self,index):
        return [self.X[index], self.Y[index]]

    def get_splits(self, t_ratio= 0.33):
        test_size = round(t_ratio * len(self.X))
        train_size = len(self.X) - test_size
        #calculate the split
        return random_split(self, [train_size, test_size])


#prepare the dataset
def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl








