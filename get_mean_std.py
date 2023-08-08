from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
import ODIR_DATASET
import CheXpert_Dataset
import pandas as pd
def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for X, _,_ in train_loader:
        for d in range(1):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train_dataset =CheXpert_Dataset.CheXpert_TEST_Dataset(neg_one_to_zero=False,transform=transform)
    print(get_mean_and_std(train_dataset))