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
                #print("batch_idx, ", batch_idx)
                # Ensure data is converted to float and moved to the correct device
                data = data.float().to(self.model.device)  # Convert data to float here
                for name, layer in self.model.named_children():
                    #initial_layer_flag =True
                    #print(f"Running layer: {name}",layer)
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

    # def extract_features(self, sample, label): #tprime version
    #     features = []
    #     labels = []
    #     i=0
    #     data= sample
    #     target = label
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

    #     with torch.no_grad():
    #         # Ensure data is converted to float and moved to the correct device
    #         data = data.float().to(self.model.device)  # Convert data to float here
    #         for name, layer in self.model.named_children():
    #             #initial_layer_flag =True
    #             print(f"Running layer: {name}",layer)
    #             #handle any special case layers to skip or transform as appropriate for your model
    #             if "LayerNorm" in name:
    #                 continue
    #             if "feature_extractor" in name:
    #                 continue
    #             if "fc1" in name:
    #                 data= torch.flatten(data, 1)
    #             data=layer(data)
                
    #             #break out after target layer reached
    #             if self.config["embedding_similarity_params"]["target_layer_to_extract"] in name:
    #                 break
        
    #         features.append(data.cpu().numpy())
    #         labels.append(target)
    #     features = np.concatenate(features, axis=0)
    #     #labels = np.concatenate(labels, axis=0)
    #     #print(labels)
    #     return features, labels
    
    def extract_features(self, sample, label): #oracle version
        import torch.nn as nn
        features = []
        labels = []
        data = sample
        target = label

        # Convert data to torch tensor and ensure it has the correct shape
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        else:
            data = data.float()

        # Move the tensor to the same device as the model
        device = next(self.model.parameters()).device
        data = data.to(device)

        # Ensure the input shape matches the model's expected dimensions
        if len(data.shape) == 2:
            data = data.unsqueeze(0)
        elif len(data.shape) == 1:
            data = data.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            # Iterate through all layers in the model
            for name, layer in self.model.named_children():
                #print(f"Running layer: {name}", layer)
                
                # Skip LayerNorm if present
                if "LayerNorm" in name:
                    continue
                
                # If the layer is feature_extractor, iterate through its sub-layers
                if "feature_extractor" in name:
                    for sub_idx, sub_layer in enumerate(layer):
                        data = sub_layer(data)
                        current_layer_name = f"{name}.{sub_idx}"
                        #print(f"Running sub layer: {current_layer_name}", sub_layer)
                        
                        # Check if the current sub-layer is the target layer
                        if self.config["embedding_similarity_params"]["target_layer_to_extract"] == current_layer_name:
                            features.append(data.cpu().numpy())
                            labels.append(target)
                            return np.concatenate(features, axis=0), labels
                else:
                    # For fully connected layers
                    if isinstance(layer, nn.Linear):
                        data = torch.flatten(data, 1)
                    data = layer(data)
                    
                    # Check if this layer is the target layer
                    if self.config["embedding_similarity_params"]["target_layer_to_extract"] == name:
                        features.append(data.cpu().numpy())
                        labels.append(target)
                        #return np.concatenate(features, axis=0), labels
                        break

            # # If target layer wasn't found inside feature_extractor
            # features.append(data.cpu().numpy())
            # labels.append(target)

        features = np.concatenate(features, axis=0)
        return features, labels
    


  
    
    def plot_tsne(self, features, labels, num_original, test_type):
        """Plots t-SNE for the given features and labels, distinguishing original from augmented samples."""
        fig, ax = plt.subplots(figsize=(10, 7))
        tsne = cuml.manifold.TSNE(n_components=2, perplexity=30, n_neighbors=120)
        tsne_results = tsne.fit_transform(features)

        # Extract points and labels
        original_points = tsne_results[:num_original, :]
        augmented_points = tsne_results[num_original:, :]
        original_labels = labels[:num_original]
        augmented_labels = labels[num_original:]

        # Define a colormap
        cmap = plt.get_cmap('tab10')

        # Calculate the number of unique labels and their respective color in the colormap
        unique_labels = np.unique(labels)
        colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]
        label_to_color = dict(zip(unique_labels, colors))

        # Settings for the plot
        edge_colors = ["white", "black"]  # Different edge colors for original and augmented
        markers = ['o', 'o']  # Same markers for original and augmented
        alpha_values = [1.0, 1.0]  # Different opacity for original and augmented

        # Plot each category with consistent color but different styles for original and augmented
        for label in unique_labels:
            idx_org = original_labels == label
            idx_aug = augmented_labels == label
            color = label_to_color[label]
            ax.scatter(original_points[idx_org, 0], original_points[idx_org, 1], 
                    color=color, marker=markers[0], edgecolor=edge_colors[0], alpha=alpha_values[0], 
                    label=f'Label {label} Original' if np.sum(idx_org) > 0 else "")
            ax.scatter(augmented_points[idx_aug, 0], augmented_points[idx_aug, 1], 
                    color=color, marker=markers[1], edgecolor=edge_colors[1], alpha=alpha_values[1], 
                    label=f'Label {label} Augmented' if np.sum(idx_aug) > 0 else "")

        ax.set_title(f't-SNE Visualization for {test_type}')
        ax.legend(loc='upper right')
        plt.show()