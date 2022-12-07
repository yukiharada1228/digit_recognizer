import numpy as np
import torch
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    
    def __init__(self, dataframe, transform):
        
        df = dataframe
        
        channel = 1
        width = 28
        height = 28
        n_pixels = channel*width*height

        if len(df.columns) == n_pixels:
            # test
            self.X = df.values.reshape(-1, 28, 28).astype(np.uint8)[..., np.newaxis]
            self.y = None
        else:
            # train
            self.X = df.drop('label', axis=1).values.reshape(-1, 28, 28).astype(np.uint8)[..., np.newaxis]
            self.y = torch.from_numpy(df['label'].values)
        
        self.tf = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is None:
            # test
            return self.tf(self.X[idx])
        else:
            # train
            return self.tf(self.X[idx]), self.y[idx]


if __name__ == '__main__':

    import pandas as pd
    from torchvision import transforms


    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    transform = transforms.Compose([transforms.ToPILImage(), 
                                    transforms.ToTensor()])
    
    train = MNISTDataset(train_df, transform=transform)
    test = MNISTDataset(test_df, transform=transform)

    X, y = next(iter(train))
    print('train')
    print(X)
    print(y)
    X = next(iter(test))
    print('test')
    print(X)
