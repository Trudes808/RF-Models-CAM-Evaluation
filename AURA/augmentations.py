import torch
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import logging

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.augmented_samples = []
        self.index_map = {}

    def add_aug_sample(self, augmented_sample, original_idx):
        # Check if sample is already augmented
        if original_idx in self.index_map:
            logging.warn("Index already augmented is this intentional?", original_idx)
            return self.index_map[original_idx]
        
        # Retrieve the original sample
        sample, label = self.original_dataset[original_idx]
        
        # Append to the augmented dataset
        self.augmented_samples.append((augmented_sample, label))
        
        # Store index mapping
        augmented_idx = len(self.augmented_samples) - 1
        self.index_map[original_idx] = augmented_idx

    
    def __getitem__(self, index):
        return self.augmented_samples[self.index_map[index]]

    def __len__(self):
        return len(self.augmented_samples)


class DataAugmentationTesting:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.original_dataset = dataloader.dataset
        self.awgn_aug_data = None
        self.awgn_aug_dataloader = None
        self.cfo_aug_data = None
        self.cfo_aug_dataloader = None


    def get_augmented_dataloader(self,dataset):
        augmented_dataloader = DataLoader(dataset, batch_size=self.dataloader.batch_size, shuffle=False)
        return augmented_dataloader


    def replace_with_noise(self, index):
        #replaces original sample with AWGN in augmented dataset at index
        
        #create awgn aug dataset
        if not hasattr(self, 'awgn_aug_data') or self.awgn_aug_data is None:
            self.awgn_aug_data = AugmentedDataset(self.original_dataset)
        original_sample, original_label = self.original_dataset[index]
        
        # Calculate power of the original sample
        power = np.mean(original_sample**2)

        #compute desired relative power
        power = power* self.config["augmentation_test_params"]["noise"]["relative_power"]
        
        # Generate random noise with the same power
        noise = np.random.randn(*original_sample.shape) * np.sqrt(power / np.mean(np.random.randn(*original_sample.shape)**2))
        
        # Replace the original sample in the augmented dataset with noise
        self.awgn_aug_data.add_aug_sample(noise, index)
        # print("before:", self.original_dataset[index])
        # print("after:", self.awgn_aug_data[index])

        #reload dataloader
        # _ = self.get_augmented_dataloader(self.awgn_aug_data)

    import numpy as np

    def add_frequency_offset(self, index):
        """
        Adds a random frequency offset within the specified range to the sample at the given index
        and replaces the original sample in the augmented dataset with this new version.
        """
        # Load the original sample and its label
        original_sample, original_label = self.original_dataset[index]
        # Combine I and Q channels to form a complex-valued array
        complex_signal = original_sample[0, :] + 1j * original_sample[1, :]
            
        #frequency params
        fs = self.config["augmentation_test_params"]["CFO"]["sample_rate"]
        lower_freq_range = self.config["augmentation_test_params"]["CFO"]["cfo_range"][0]
        upper_freq_range = self.config["augmentation_test_params"]["CFO"]["cfo_range"][1]
        
        # Generate a random frequency offset within the given range
        freq_offset = np.random.uniform(lower_freq_range, upper_freq_range)
        
        # Create time vector
        Ts = 1 / fs  # Calculate sample period
        t = np.arange(original_sample.shape[1]) * Ts  # Create time vector
        
        # Apply the frequency offset by multiplying the original signal with a complex exponential
        freq_offset_signal = complex_signal * np.exp(1j * 2 * np.pi * freq_offset * t)
        #print(freq_offset_signal)
        # Split the complex-valued signal back into I and Q channels
        offset_sample = np.vstack((freq_offset_signal.real, freq_offset_signal.imag))

        # Check if the augmented dataset exists, create if not
        if not hasattr(self, 'cfo_aug_data') or self.cfo_aug_data is None:
            self.cfo_aug_data = AugmentedDataset(self.original_dataset)
        #print("before:", self.original_dataset[index])
        #print(offset_sample)
        
        # Replace the original sample in the augmented dataset with the frequency offset version
        self.cfo_aug_data.add_aug_sample(offset_sample, index)
        #print("after:", self.cfo_aug_data.__getitem__(index))

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
            # print(f"Aug outputs shape: {output.shape}")
            # print(f"Aug prediction: {output}")
        
        return output
    
    def run_model_on_original_sample(self, sample):
        # Convert numpy array to torch tensor
        og_tensor = torch.from_numpy(sample).float()

        # Check the device of the model
        device = next(self.model.parameters()).device
        #print(f"Model is on device: {device}")

        # Move the tensor to the same device as the model
        og_tensor = og_tensor.to(device)

        # Ensure the input shape matches the model's expected dimensions
        if len(og_tensor.shape) == 2:
            og_tensor = og_tensor.unsqueeze(0)

        # Run the model inference
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self.model(og_tensor).detach().cpu().numpy()
            # print(f"Aug outputs shape: {output.shape}")
            # print(f"Aug prediction: {output}")
        
        return output

    def get_original_sample(self, index):
        sample, label = self.dataloader.dataset[index]
        return sample, label

    # def get_augmented_sample(self, index):
    #     sample, label = self.augmented_dataloader.dataset[self.augmented_dataset.index_map[index]]
    #     return sample, label

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
