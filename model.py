import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet1D(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet1D, self).__init__()
        list = []
        list.append(nn.Conv1d(2, 128, 7, padding="same"))
        list.append(nn.ReLU())
        list.append(nn.Conv1d(128, 128, 5, padding="same"))
        list.append(nn.ReLU())
        list.append(nn.MaxPool1d(2))
        for _ in range(6):
            list.append(nn.Conv1d(128, 128, 7, padding="same"))
            list.append(nn.ReLU())
            list.append(nn.Conv1d(128, 128, 5, padding="same"))
            list.append(nn.ReLU())
            list.append(nn.MaxPool1d(2))
        self.feature_extractor = nn.ModuleList(list)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        for layer in self.feature_extractor:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        return x


if __name__ == "__main__":
    from dataset_lora import InDistributionTrainDataset, DataLoader
    dataset = InDistributionTrainDataset()
    dataloader = DataLoader(dataset, 10, shuffle=True, num_workers=1)
    model = AlexNet1D(dataset.num_classes)
    for data, label in dataloader:
        print(data.shape, label.shape)
        print(model(data).shape)
        break
