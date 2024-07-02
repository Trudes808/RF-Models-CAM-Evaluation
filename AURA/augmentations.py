import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.augmented_samples = []
        self.index_map = {}

    def add_aug_sample(self, augmented_sample, original_idx):
        # Check if sample is already augmented
        if original_idx in self.index_map:
            return self.index_map[original_idx]
        
        # Retrieve the original sample
        sample, label = self.original_dataset[original_idx]
        
        # Append to the augmented dataset
        self.augmented_samples.append((augmented_sample, label))
        
        # Store index mapping
        augmented_idx = len(self.augmented_samples) - 1
        self.index_map[original_idx] = augmented_idx


    def __getitem__(self, index):
        return self.augmented_samples[index]

    def __len__(self):
        return len(self.augmented_samples)


class DataAugmentationTesting:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.original_dataset = dataloader.dataset
        self.augmented_dataset = AugmentedDataset(self.original_dataset)
        self.augmented_dataloader = None
    #     self.prepare_datasets()

    # def prepare_datasets(self):
    #     self.original_dataset = self.dataloader.dataset
        
    #     # # Assuming dataloader already provides a dataset. Dataset expected to be two channels first channel I second channel Q.
    #     # # Make a deep copy of the original data for manipulation, do this only as neccisary as it can be slow to copy whole thing
    #     self.augmented_dataset = CloneDataset(self.original_dataset)
    #     self.augmented_dataloader = DataLoader(self.augmented_dataset, batch_size=self.dataloader.batch_size, shuffle=False)
    #     # self.original_data = [data.clone().detach() for data, _ in self.dataloader]
    #     # self.augmented_data = self.original_data.copy()
    def get_augmented_dataloader(self):
        self.augmented_dataloader = DataLoader(self.augmented_dataset, batch_size=self.dataloader.batch_size, shuffle=False)
        return self.augmented_dataloader


    def replace_with_noise(self, index):
        #replaces original sample with AWGN in augmented dataset at index
        # Retrieve original sample
        original_sample, original_label = self.original_dataset[index]
        
        # Calculate power of the original sample
        power = np.mean(original_sample**2)
        
        # Generate random noise with the same power
        noise = np.random.randn(*original_sample.shape) * np.sqrt(power / np.mean(np.random.randn(*original_sample.shape)**2))
        
        # Replace the original sample in the augmented dataset with noise
        self.augmented_dataset.add_aug_sample(noise, index)
        print("before:", self.original_dataset[index])
        print("after:", self.augmented_dataset[self.augmented_dataset.index_map[index]])
        #reload dataloader
        _ = self.get_augmented_dataloader()

    def run_model_on_augmented_sample(self, augmented_sample):
 
        # Convert numpy array to torch tensor
        aug_tensor = torch.from_numpy(augmented_sample).float()

        # Check the device of the model
        device = next(self.model.parameters()).device
        #print(f"Model is on device: {device}")

        # Move the tensor to the same device as the model
        aug_tensor = aug_tensor.to(device)

        # Ensure the input shape matches the model's expected dimensions
        if len(aug_tensor.shape) == 2:
            aug_tensor = aug_tensor.unsqueeze(0)

        # Run the model inference
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self.model(aug_tensor).detach().cpu().numpy()
            print(f"Aug outputs shape: {output.shape}")
            print(f"Aug prediction: {output}")
        
        return output
    
    def run_model_on_original_sample(self, index):
        # Load the augmented sample
        original_sample = self.original_dataset[index]
        
        # Assuming the model expects a batch dimension
        original_sample = original_sample.unsqueeze(0)  # Add batch dimension if necessary
        
        # Run the model inference
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self.model(original_sample)
        
        return output

    def get_original_sample(self, index):
        sample, label = self.dataloader.dataset[index]
        return sample, label

    def get_augmented_sample(self, index):
        sample, label = self.augmented_dataloader.dataset[self.augmented_dataset.index_map[index]]
        return sample, label

# Modular Test Usage
if __name__ == "__main__":
    # Mock-up PyTorch model and dataloader
    model = torch.nn.Linear(2, 2)  # Placeholder for the actual pretrained model
    dataloader = torch.utils.data.DataLoader(torch.randn(100, 2, 2), batch_size=1)
    config_path = 'path_to_config.json'

    augmentor = DataAugmentationTesting(model, dataloader, config_path)
    augmentor.augment_with_noise(0)  # Augment the first sample with noise
    output = augmentor.run_model_on_augmented_sample(0)  # Run inference on the augmented sample
    print("Model Output on Augmented Sample:", output)
    print("Original Sample:", augmentor.get_original_sample(0))
    print("Augmented Sample:", augmentor.get_augmented_sample(0))
