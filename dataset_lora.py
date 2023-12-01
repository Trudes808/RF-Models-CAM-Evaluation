import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from scipy.io import loadmat

FRACTION = .1


class InDistributionTrainDataset(Dataset):
    def __init__(self, seqs, seq_len=256, data_dir="/raid/backup_storage_oldDGX/LORA/Year_1_outdoor/outdoor_dataset_1/mat_files/raw/"):
        with open(data_dir + "train.pkl", "rb") as f:
            file_dict = pickle.load(f)
        filenames = file_dict["files"]
        self.sequences = []
        self.idxs = [0]
        for name in filenames:
            print(len(self.sequences), len(filenames))
            seq = seqs[name]
            seq = seq[:int(.6 * len(seq) * FRACTION)]
            real = np.real(seq) * 1000
            # real -= np.mean(real)
            # real /= np.std(real)
            imag = np.imag(seq) * 1000
            # imag -= np.mean(imag)
            # imag /= np.std(imag)
            self.sequences.append(np.copy(np.stack((real, imag), axis=0)))
            self.idxs.append(self.idxs[-1] + len(self.sequences[-1][0]) - seq_len + 1)
        self.labels = file_dict["labels"]
        self.num_classes = max(self.labels) + 1
        self.seq_len = seq_len

    def __len__(self):
        return sum([len(seq[0]) for seq in self.sequences]) - len(self.sequences) * (self.seq_len - 1)

    def __getitem__(self, item):
        i, idx = self.__get_idx(item)
        return self.sequences[i][:, idx: idx + self.seq_len], self.labels[i]

    def __get_idx(self, item):
        for i, idx in enumerate(self.idxs):
            if item < idx:
                return i - 1, item - self.idxs[i - 1]


class WildDataset(Dataset):
    def __init__(self, id_seqs, ood_seqs, seq_len=256,
                 data_dir="/raid/backup_storage_oldDGX/LORA/Year_1_outdoor/outdoor_dataset_1/mat_files/raw/"):
        self.sequences = []
        self.idxs = [0]

        with open(data_dir + "train.pkl", "rb") as f:
            file_dict = pickle.load(f)
        filenames = file_dict["files"]
        for name in filenames:
            print(len(self.sequences), len(filenames))
            seq = id_seqs[name]
            seq = seq[int(.9 * len(seq) * FRACTION): int(.91 * len(seq) * FRACTION)]
            real = np.real(seq) * 1000
            imag = np.imag(seq) * 1000
            self.sequences.append(np.copy(np.stack((real, imag), axis=0)))
            self.idxs.append(self.idxs[-1] + len(self.sequences[-1][0]) - seq_len + 1)

        with open(data_dir + "ood_wild.pkl", "rb") as f:
            file_dict = pickle.load(f)
        filenames = file_dict["files"]
        for name in filenames:
            print(len(self.sequences), len(filenames))
            seq = ood_seqs[name]
            seq = seq[:int(len(seq) * FRACTION)]
            real = np.real(seq) * 1000
            imag = np.imag(seq) * 1000
            self.sequences.append(np.copy(np.stack((real, imag), axis=0)))
            self.idxs.append(self.idxs[-1] + len(self.sequences[-1][0]) - seq_len + 1)

        self.seq_len = seq_len
        self.full_size = sum([len(seq[0]) for seq in self.sequences]) - len(self.sequences) * (self.seq_len - 1)
        self.rnd = None

    def __len__(self):
        return self.full_size // 10

    def __getitem__(self, item):
        if self.rnd is None:
            self.rnd = np.random.RandomState(item)
        item = self.rnd.randint(self.full_size)
        i, idx = self.__get_idx(item)
        return self.sequences[i][:, idx: idx + self.seq_len]

    def __get_idx(self, item):
        for i, idx in enumerate(self.idxs):
            if item < idx:
                return i - 1, item - self.idxs[i - 1]


class InDistributionTestDataset(Dataset):
    def __init__(self, seqs, split, seq_len=256, example_len=450,
                 data_dir="/raid/backup_storage_oldDGX/LORA/Year_1_outdoor/outdoor_dataset_1/mat_files/raw/"):
        with open(data_dir + "train.pkl", "rb") as f:
            file_dict = pickle.load(f)
        filenames = file_dict["files"]
        self.sequences = []
        self.idxs = [0]
        labels = file_dict["labels"]
        self.labels = []
        for name, label in zip(filenames, labels):
            # print(len(self.sequences), len(filenames))
            seq = seqs[name]
            if split == "train":
                seq = seq[:int(.6 * len(seq) * FRACTION)]
            elif split == "val":
                seq = seq[int(.6 * len(seq) * FRACTION): int(.7 * len(seq) * FRACTION)]
            else:
                seq = seq[int(.7 * len(seq) * FRACTION): int(.9 * len(seq) * FRACTION)]
            #for i in range(len(seq) // tran_len):
                #transmission = seq[i * tran_len: (i + 1) * tran_len]
            if len(seq)/seq_len < example_len:
                example_len = len(seq)/seq_len
                print("WARNING: Not enough sequences available in file for desired example len. Setting example len to number of seqs From", name, " with #seqs=", len(seq)/seq_len)
            for i in range(0,len(seq),seq_len):
                if i/seq_len > example_len:
                    break
                transmission = seq[i: (i + seq_len)]
                real = np.real(transmission) * 1000
                # real -= np.mean(real)
                # real /= np.std(real)
                imag = np.imag(transmission) * 1000
                # imag -= np.mean(imag)
                # imag /= np.std(imag)
                self.sequences.append(np.copy(np.stack((real, imag), axis=0)))
                self.idxs.append(self.idxs[-1] + len(real) - seq_len + 1)
                self.labels.append(label)
        self.num_classes = max(self.labels) + 1
        self.seq_len = seq_len
        self.example_len = example_len
        print("Loaded Test Dataset of size", len(self.sequences), self.example_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        data = []
        for idx in range(self.example_len - self.seq_len + 1):
            data.append(np.copy(self.sequences[item][:, idx: idx + self.seq_len]))
        data = np.stack(data, axis=0)
        return data, np.ones(self.example_len - self.seq_len + 1) * self.labels[item]


class OutOfDistributionDataset(Dataset):
    def __init__(self, split, seen=True, seq_len=256, tran_len=2000,
                 data_dir="/raid/backup_storage_oldDGX/LORA/Year_1_outdoor/outdoor_dataset_1/mat_files/raw/"):
        with open(data_dir + "ood_%s.pkl" % split, "rb") as f:
            file_dict = pickle.load(f)
        filenames = file_dict["files"]
        seqs = {}
        for i, name in enumerate(filenames):
            print(i, len(filenames))
            seqs[name] = loadmat(name)["f_sig"][0]
        self.sequences = []
        self.idxs = [0]
        labels = file_dict["labels"]
        self.labels = []
        for name, label in zip(filenames, labels):
            seq = seqs[name]
            seq = seq[:int(len(seq) * FRACTION * 0.1)] if seen else seq[int(len(seq) * FRACTION): int(
                len(seq) * FRACTION * 1.1)]
            for i in range(len(seq) // tran_len):
                transmission = seq[i * tran_len: (i + 1) * tran_len]
                real = np.real(transmission) * 1000
                # real -= np.mean(real)
                # real /= np.std(real)
                imag = np.imag(transmission) * 1000
                # imag -= np.mean(imag)
                # imag /= np.std(imag)
                self.sequences.append(np.copy(np.stack((real, imag), axis=0)))
                self.idxs.append(self.idxs[-1] + len(real) - seq_len + 1)
                self.labels.append(label)
        self.num_classes = max(self.labels) + 1
        self.seq_len = seq_len
        self.example_len = tran_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        data = []
        for idx in range(self.example_len - self.seq_len + 1):
            data.append(np.copy(self.sequences[item][:, idx: idx + self.seq_len]))
        data = np.stack(data, axis=0)
        return data, np.ones(self.example_len - self.seq_len + 1) * self.labels[item]


if __name__ == "__main__":
    dataset = InDistributionTrainDataset()
    print(len(dataset))
    # dataloader = DataLoader(dataset, 10, shuffle=True, num_workers=1)

    # dataset = InDistributionTestDataset("val")
    # dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=1)
    #
    # for data, label in dataloader:
    #     print(data, label)
    #     print(data.shape, label.shape)
    #     break
