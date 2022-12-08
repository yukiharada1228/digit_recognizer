import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


logger = logging.getLogger(__name__)


class MNISTDataset(Dataset):

    def __init__(self, dataframe, transform=transforms.ToTensor()):

        self.df = dataframe

        channel = 1
        width = 28
        height = 28
        n_pixels = channel*width*height

        if len(self.df.columns) == n_pixels:
            self.X, self.y = self.test()
        else:
            self.X, self.y = self.train()

        self.tf = transform

    def train(self):
        X = self.df.drop('label', axis=1).values.reshape(-1, 28, 28)
        X = X.astype(np.uint8)[..., np.newaxis]
        y = torch.from_numpy(self.df['label'].values)
        logger.debug(
            {'action': 'train', 'X.shape': X.shape, 'y.size()': y.size()})
        return X, y

    def test(self):
        X = self.df.values.reshape(-1, 28, 28)
        X = X.astype(np.uint8)[..., np.newaxis]
        y = None
        logger.debug({'action': 'test', 'X.shape': X.shape, 'y': y})
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            # test
            X = self.tf(self.X[idx])
            logger.debug({'action': '__getitem__', 'X.size()': X.size()})
            return X
        else:
            # train
            X, y = self.tf(self.X[idx]), self.y[idx]
            logger.debug({'action': '__getitem__',
                          'X.size()': X.size(),
                          'y.size()': y.size()})
            return X, y


if __name__ == '__main__':

    import sys

    import pandas as pd
    from torch.utils.data import DataLoader

    logging.basicConfig(level=logging.DEBUG,
                        stream=sys.stdout)

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    train = MNISTDataset(train_df)
    test = MNISTDataset(test_df)

    batch_size = 3
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))
    logger.debug({'X': X.size(), 'y': y.size()})
    X = next(iter(test_loader))
    logger.debug({'X': X.size()})
