from sklearn.manifold import TSNE
import itertools
import cuml
import torch
import numpy as np
import matplotlib.pyplot as plt




class EmbeddingSimilarityModule:

    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        #self.prepare_datasets()


    # Function to extract features from the last fully connected layer before softmax
    def extract_features_batch(self, test_idx=None):
        features = []
        labels = []
        i=0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.dataloader):
                batch_size = len(data)
                print("batch_idx, ", batch_idx)
                # Ensure data is converted to float and moved to the correct device
                data = data.float().to(self.model.device)  # Convert data to float here
                for name, layer in self.model.named_children():
                    #initial_layer_flag =True
                    print(f"Running layer: {name}",layer)
                    #handle any special case layers to skip or transform as appropriate for your model
                    if "LayerNorm" in name:
                        continue
                    if "feature_extractor" in name:
                        continue
                    if "fc1" in name:
                        data= torch.flatten(data, 1)
                    data=layer(data)
                    
                    #break out after target layer reached
                    if self.config["embedding_similarity_params"]["target_layer_to_extract"] in name:
                        break
            
                features.append(data.cpu().numpy())
                labels.append(target.numpy())
                if i >= self.config["embedding_similarity_params"]["n_batches_to_test"]:
                    break
                i+=1
        # #run augmented sample as well if available
        # if aug_idx is not None:
        #     data, target = self.dataloader.dataset[aug_idx]
        #     # Convert data to torch tensor and ensure it has the correct shape
        #     if isinstance(data, np.ndarray):
        #         data = torch.from_numpy(data).float()
        #     else:
        #         data = data.float()

        #     # Check the device of the model
        #     device = next(self.model.parameters()).device
        #     # Move the tensor to the same device as the model
        #     data = data.to(device)

        #     # Ensure the input shape matches the model's expected dimensions
        #     if len(data.shape) == 2:
        #         data = data.unsqueeze(0)
        #     elif len(data.shape) == 1:
        #         data = data.unsqueeze(0).unsqueeze(0)

        #     # Get the output from the last fully connected layer (fc1)
        #     x = self.model.conv1(data)
        #     x = self.model.relu1(x)
        #     x = self.model.maxpool1(x)
        #     x = self.model.conv2(x)
        #     x = self.model.relu2(x)
        #     x = self.model.maxpool2(x)
        #     x = torch.flatten(x, 1)
        #     x = self.model.fc1(x)
        #     x = self.model.relu3(x)
        #     # x = model.fc2(x)
        #     # x = model.logSoftmax(x)
            
        #     features.append(x.cpu().detach().numpy())
        #     labels.append(target.numpy() if isinstance(target, torch.Tensor) else np.array([target]))

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels

    def extract_features(self, sample, label):
        features = []
        labels = []
        i=0
        data= sample
        target = label
        # Convert data to torch tensor and ensure it has the correct shape
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        else:
            data = data.float()

        # Check the device of the model
        device = next(self.model.parameters()).device
        # Move the tensor to the same device as the model
        data = data.to(device)

        # Ensure the input shape matches the model's expected dimensions
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        elif len(data.shape) == 1:
            data = data.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Ensure data is converted to float and moved to the correct device
            data = data.float().to(self.model.device)  # Convert data to float here
            for name, layer in self.model.named_children():
                #initial_layer_flag =True
                print(f"Running layer: {name}",layer)
                #handle any special case layers to skip or transform as appropriate for your model
                if "LayerNorm" in name:
                    continue
                if "feature_extractor" in name:
                    continue
                if "fc1" in name:
                    data= torch.flatten(data, 1)
                data=layer(data)
                
                #break out after target layer reached
                if self.config["embedding_similarity_params"]["target_layer_to_extract"] in name:
                    break
        
            features.append(data.cpu().numpy())
            labels.append(target)
        features = np.concatenate(features, axis=0)
        #labels = np.concatenate(labels, axis=0)
        print(labels)
        return features, labels


    def plot_tsne(self,features,labels):
        fig, ax = plt.subplots(1, 3, figsize=(20,7), constrained_layout=True)
        for c, per in zip(itertools.count(), [5, 30, 50]):
            tsne = cuml.manifold.TSNE(n_components=2,
                        perplexity=per,
                        n_neighbors=per*4)
            tsne = tsne.fit_transform(features)
            scatter = ax[c].scatter(tsne[:-1, 0], tsne[:-1, 1], c=labels[:-1], cmap='tab10', s=0.3)
            scatter = ax[c].scatter(tsne[-1, 0], tsne[-1, 1], c=labels[-1], cmap='tab10',marker='*', edgecolors='k', s=50)
            ax[c].set_title(f'Perplexity: {per}', fontsize=16)    

        fig.suptitle('t-SNE Dimensionality reduction', fontweight='bold', fontsize=25)
        cbar = fig.colorbar(scatter, boundaries=np.arange(11)-0.5, location='right')
        cbar.set_ticks(np.arange(10))
        cbar.set_ticklabels(np.arange(10))
        plt.show()