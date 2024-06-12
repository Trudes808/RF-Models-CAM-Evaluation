import torch
import numpy as np
import json

class DataAugmentationTesting:
    def __init__(self, model, dataloader, config_path):
        self.model = model
        self.dataloader = dataloader
        self.config = self.load_config(config_path)
        self.original_data = None
        self.augmented_data = None
        self.prepare_datasets()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config

    def prepare_datasets(self):
        # Assuming dataloader already provides a dataset. Dataset expected to be two channels first channel I second channel Q.
        # Make a deep copy of the original data for manipulation
        self.original_data = [data.clone().detach() for data, _ in self.dataloader]
        self.augmented_data = self.original_data.copy()

    def augment_with_noise(self, index):
        # Retrieve original sample
        original_sample = self.original_data[index]
        
        # Calculate power of the original sample
        power = torch.mean(original_sample**2)
        
        # Generate random noise with the same power
        noise = torch.randn_like(original_sample) * torch.sqrt(power / torch.mean(torch.randn_like(original_sample)**2))
        
        # Replace the original sample in the augmented dataset with noise
        self.augmented_data[index] = noise

    def run_model_on_augmented_sample(self, index):
        # Load the augmented sample
        augmented_sample = self.augmented_data[index]
        
        # Assuming the model expects a batch dimension
        augmented_sample = augmented_sample.unsqueeze(0)  # Add batch dimension if necessary
        
        # Run the model inference
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            output = self.model(augmented_sample)
        
        return output

    def get_original_sample(self, index):
        return self.original_data[index]

    def get_augmented_sample(self, index):
        return self.augmented_data[index]

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
