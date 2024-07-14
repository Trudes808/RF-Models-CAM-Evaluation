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
        self.snr_aug_data = None
        self.snr_aug_dataloader = None
        self.fspl_aug_data = None
        self.fspl_aug_dataloader = None
        self.phase_rot_data = None
        self.phase_rot_dataloader = None
        self.iq_aug_data = None
        self.iq_aug_dataloader = None


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
        original_power = np.mean(original_sample**2)

        #compute desired relative power
        # Get the relative power factor from config in dB and convert it to linear scale
        relative_power_dB = self.config["augmentation_test_params"]["noise"]["relative_power"]
        relative_power_linear = 10 ** (relative_power_dB / 10)
        
        # Calculate the absolute noise power
        noise_power = original_power * relative_power_linear
        
        # Generate random noise with calculated relative power
        noise = np.random.randn(*original_sample.shape)
        # Normalize the noise to have the desired power
        noise = noise * np.sqrt(noise_power / np.mean(noise**2))
        
        # Replace the original sample in the augmented dataset with the noise
        self.awgn_aug_data.add_aug_sample(noise, index)


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
        lower_freq_range = self.config["augmentation_test_params"]["CFO"]["cfo_range_Hz"][0]
        upper_freq_range = self.config["augmentation_test_params"]["CFO"]["cfo_range_Hz"][1]
        
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
    
    def reduce_SNR(self, index):
        """
        Reduces the SNR of the sample at the given index to the level specified in the config.
        SNR positive values mean the signal is above the noise level by that many dB.
        Negative SNR values mean the noise is above the signal level by that many dB.
        """
        original_sample, original_label = self.original_dataset[index]
        desired_SNR_dB = self.config["augmentation_test_params"]["SNR"]["desired_SNR_dB"]

        # Combine I and Q into a complex signal and compute its power
        complex_signal = original_sample[0, :] + 1j * original_sample[1, :]
        signal_power = np.mean(np.abs(complex_signal)**2)

        # Calculate the required noise power to achieve the desired SNR
        desired_SNR_linear = 10**(desired_SNR_dB / 10)
        noise_power = signal_power / desired_SNR_linear

        # Generate noise with the appropriate power and add it to the signal
        noise = np.random.normal(0, np.sqrt(noise_power / 2), size=complex_signal.shape) + \
                1j * np.random.normal(0, np.sqrt(noise_power / 2), size=complex_signal.shape)
        noisy_signal = complex_signal + noise

        # Split the complex signal back into I and Q channels
        noisy_sample = np.vstack((noisy_signal.real, noisy_signal.imag))

        # Update the augmented dataset
        if not hasattr(self, 'snr_aug_data') or self.snr_aug_data is None:
            self.snr_aug_data = AugmentedDataset(self.original_dataset)
        
        self.snr_aug_data.add_aug_sample(noisy_sample, index)
    
    def apply_FSPL(self, index):
        """
        Applies Free Space Path Loss (FSPL) to the sample at the given index based on the distance range
        and path loss exponent specified in the configuration. Ensures that distance is never zero. Uses simplified FSPL to 
        ignore frequency of transmission and improve calculation efficiency.
        """
        # Load the original sample and its label
        original_sample, original_label = self.original_dataset[index]

        # Retrieve FSPL configuration parameters
        distance_min, distance_max = self.config["augmentation_test_params"]["FSPL"]["distance_range_m"]
        alpha = self.config["augmentation_test_params"]["FSPL"]["alpha"]
        
        # Ensure the minimum distance is greater than zero to avoid division by zero
        if distance_min <= 0:
            distance_min = 0.1  # Set a small but non-zero minimum distance

        # Generate a random distance within the specified range
        distance = np.random.uniform(distance_min, distance_max)
        
        # Calculate FSPL attenuation factor using the simplified model of 1/d^alpha
        attenuation_factor = 1 / (distance ** alpha)
        
        # Apply attenuation to the original sample
        attenuated_sample = original_sample * attenuation_factor
        
        # Check if the FSPL augmented dataset exists, create if not
        if not hasattr(self, 'fspl_aug_data') or self.fspl_aug_data is None:
            self.fspl_aug_data = AugmentedDataset(self.original_dataset)
        
        # Replace the original sample in the augmented dataset with the attenuated version
        self.fspl_aug_data.add_aug_sample(attenuated_sample, index)
    def apply_phase_rotation(self, index):
        """
        Applies a random phase rotation to the sample at the given index based on the angle range specified in the configuration.
        """
        # Load the original sample and its label
        original_sample, original_label = self.original_dataset[index]

        # Combine I and Q channels to form a complex-valued array
        complex_signal = original_sample[0, :] + 1j * original_sample[1, :]
        
        # Retrieve phase rotation configuration parameters
        min_angle, max_angle, step_angle = self.config["augmentation_test_params"]["phase_rotation"]["angle_range_deg"]
        
        # Generate a random angle within the specified range
        angle_deg = np.random.choice(np.arange(min_angle, max_angle + step_angle, step_angle))
        angle_rad = np.deg2rad(angle_deg)  # Convert angle from degrees to radians
        
        # Calculate the phase rotation factor
        rotation_factor = np.exp(1j * angle_rad)
        
        # Apply the phase rotation to the signal
        rotated_signal = complex_signal * rotation_factor
        
        # Split the complex-valued signal back into I and Q channels
        rotated_sample = np.vstack((rotated_signal.real, rotated_signal.imag))

        # Check if the phase rotation augmented dataset exists, create if not
        if not hasattr(self, 'phase_rot_aug_data') or self.phase_rot_aug_data is None:
            self.phase_rot_aug_data = AugmentedDataset(self.original_dataset)
        
        # Replace the original sample in the augmented dataset with the rotated version
        self.phase_rot_aug_data.add_aug_sample(rotated_sample, index)
    
    def apply_IQ_imbalance(self, index):
        """
        Applies IQ imbalance to the sample at the given index based on the gain and phase imbalance parameters specified in the configuration.
        """
        original_sample, original_label = self.original_dataset[index]

        # Retrieve IQ imbalance configuration parameters
        gain_imbalance = self.config["augmentation_test_params"]["IQ_imbalance"]["gain_factor"]
        phase_imbalance = np.deg2rad(self.config["augmentation_test_params"]["IQ_imbalance"]["phase_deg"])  # Convert degrees to radians

        # Apply IQ imbalance
        I = original_sample[0, :]
        Q = original_sample[1, :]

        I_prime = gain_imbalance * I + phase_imbalance * Q
        Q_prime = phase_imbalance * I + gain_imbalance * Q

        imbalanced_sample = np.vstack((I_prime, Q_prime))

        # Check if the IQ imbalance augmented dataset exists, create if not
        if not hasattr(self, 'iq_aug_data') or self.iq_aug_data is None:
            self.iq_aug_data = AugmentedDataset(self.original_dataset)
        
        self.iq_aug_data.add_aug_sample(imbalanced_sample, index)



    # def run_model_on_augmented_sample(self, augmented_sample):
 
    #     # Convert numpy array to torch tensor
    #     aug_tensor = torch.from_numpy(augmented_sample).float()

    #     # Check the device of the model
    #     device = next(self.model.parameters()).device
    #     #print(f"Model is on device: {device}")

    #     # Move the tensor to the same device as the model
    #     aug_tensor = aug_tensor.to(device)

    #     # Ensure the input shape matches the model's expected dimensions
    #     if len(aug_tensor.shape) == 2:
    #         aug_tensor = aug_tensor.unsqueeze(0)

    #     # Run the model inference
    #     self.model.eval()  # Set the model to evaluation mode
    #     with torch.no_grad():
    #         output = self.model(aug_tensor).detach().cpu().numpy()
    #         # print(f"Aug outputs shape: {output.shape}")
    #         # print(f"Aug prediction: {output}")
        
    #     return output
    
    def run_model_on_sample(self, sample):
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
