from torch.utils.data import Dataset, DataLoader
import torch

torch.manual_seed(0)

def shuffle_data(data, args):
    train_portion = 0.9
    n_samps = len(data)
    n_train = int(train_portion * n_samps)
    inds_train = data[:n_train]
    inds_test = data[n_train:]
    train_data = data[inds_train]
    test_data = data[inds_test]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    return train_data, test_data, train_loader, test_loader

