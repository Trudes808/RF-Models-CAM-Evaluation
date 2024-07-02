import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os


def load_pretrained_model(config):
    # ***********************************************
    # Placeholder: Load your model here
    # This should be replaced with the specific model loading mechanism
    # model = torch.nn.Linear(2048, 10)  # Example: simple linear model with 2048 input features
    #*****************************************************

    #*************************************************** t-prime example
    from model_under_test.model import Baseline_CNN1D
    #t-prime
    pretrained_model_path = config["pretrained_model_path"]
    #PATH = './results/t-prime/SNR30/'
    Nclass = 4
    num_channels = 2
    num_feats = 1
    slice_len = 512

    device = torch.device('cpu')
    # print(device)
    # cur_dir = os.getcwd()

    model = Baseline_CNN1D(classes=Nclass, numChannels=num_channels, slice_len=slice_len)
    checkpoint = torch.load(pretrained_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) # reload the model on the appropriate device
    model.device = device
    model.eval()    # set the evaluation mode
    print(model,next(model.parameters()).device)
    print(model.device)
    #***************************************

    return model


def load_dataloader(config):
    # ***********************************************
    # Placeholder: Load your dataloader here
    # This should be replaced with the specific model dataloader loading mechanism
    #*****************************************************

    #*************************************************** t-prime example
    from model_under_test.dataset_tprime import TPrimeDataset
    import argparse

    #TODO: Change from hardcode to config read
    args=argparse.Namespace()
    args.protocols = ['802_11ax', '802_11b_upsampled', '802_11n', '802_11g']
    args.noise = True
    args.snr_db = [30]
    args.raw_path = "/home/sagetrudeau/Projects/t-prime/data/DATASET1_1"
    args.slicelen = 512
    args.overlap_ratio = 0.0
    args.postfix = ''
    args.raw_data_ratio = 1.0
    args.channel = None
    args.out_mode = 'real'
    args.worker_batch_size = 512

    ds_test = TPrimeDataset(args.protocols,
                          ds_path=args.raw_path,
                          ds_type='test',
                          snr_dbs=args.snr_db,
                          slice_len=args.slicelen,
                          slice_overlap_ratio=float(args.overlap_ratio),
                          raw_data_ratio=args.raw_data_ratio,
                          file_postfix=args.postfix,
                          override_gen_map=False,    # it will use the same as above call
                          apply_wchannel=args.channel,
                          apply_noise=args.noise,
                          out_mode=args.out_mode)
    dataloader = DataLoader(ds_test, batch_size=args.worker_batch_size, shuffle=False)
    dataloader_aug = DataLoader(ds_test, batch_size=args.worker_batch_size, shuffle=False)
    print(dataloader.dataset[0])
    return dataloader

def analyze_model_input_shape(model):
    # This function assumes the model's first layer can give us the input shape
    # Depending on the model architecture, you might need to adapt this
    try:
        input_shape = model.input_shape  # Custom attribute or predefined
    except AttributeError:
        # Assuming the first layer is accessible and its weight can tell the input shape
        input_layer = next(model.children())
        input_shape = input_layer.weight.shape[1]  # Linear layers have shape [out_features, in_features]
    return input_shape

def transform_dataloader(dataloader, input_shape):
    # Transform data to have the correct input shape
    transformed_samples = []
    labels = []

    for data, label in dataloader:
        # Assuming data is initially (batch_size, channels, length)
        # and needs to be transformed to (batch_size, slices, channels, model_input_length)
        batch_size, channels, length = data.shape
        model_input_length = input_shape // channels  # Since model expects flat input per sample

        # Reshape data to fit model input
        if length != model_input_length:
            # Handle case where length of data is not what model expects
            # This example simply truncates or pads the data
            if length > model_input_length:
                data = data[:, :, :model_input_length]  # Truncate
            else:
                padding = model_input_length - length
                data = torch.nn.functional.pad(data, (0, padding), "constant", 0)  # Pad

        # Store transformed data
        transformed_samples.append(data)
        labels.append(label)

    # Re-create dataloader with transformed dataset
    transformed_dataset = TensorDataset(torch.cat(transformed_samples, dim=0), torch.cat(labels, dim=0))
    transformed_dataloader = DataLoader(transformed_dataset, batch_size=dataloader.batch_size)

    return transformed_dataloader

# Example usage
if __name__ == "__main__":
    # Example dataloader with random data
    original_dataloader = DataLoader(TensorDataset(torch.randn(100, 2, 3000), torch.randint(0, 10, (100,))), batch_size=10)

    # Load model and analyze input shape
    model = load_pretrained_model()
    input_shape = analyze_model_input_shape(model)

    # Transform the dataloader to match model input requirements
    transformed_dataloader = transform_dataloader(original_dataloader, input_shape)
