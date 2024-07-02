import numpy as np
from model_loader import load_pretrained_model, load_dataloader
from augmentations import DataAugmentation
from embeddings_similarity_module import EmbeddingSimilarity
from cam_module import ClassActivationMapping
from collections import defaultdict
import random
import json
import os

def load_config(config_path):
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def save_test_results(results):
    # Implement the logic to save results to a file or database
    pass

def print_summary(results):
    # Summarize test results
    print("Test Summary:")
    for result in results:
        print(f"Index: {result['index']}, Model Output: {result['output']}, Embedding Similarity: {result['embedding_similarity']}, CAM Result: {result['cam_result']}")

#***************************************************      Main           *************************************************************
if __name__ == "__main__":

    config_path = "config.json"
    config = load_config(config_path)
    model = load_pretrained_model()
    dataloader = load_dataloader()

    # Initialize the Data Augmentation Module
    augmentor = DataAugmentation(model, dataloader, config_path)
    
    # Initialize processing modules
    embedding_similarity = EmbeddingSimilarity(model)
    cam = ClassActivationMapping(model)

    # Generate test index range
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataloader.dataset):
        class_indices[label.item()].append(idx)

    N = 3  # Number of samples per class
    test_index_range = []
    for indices in class_indices.values():
        if len(indices) >= N:
            test_index_range.extend(random.sample(indices, N))
        else:
            test_index_range.extend(indices)

    # Shuffle the test index range to ensure random order
    random.shuffle(test_index_range)

    # Prepare to collect results
    test_results = []

    # Run tests on the specific range of indexes
    for index in test_index_range:
        # Augment the sample
        augmentor.augment_with_noise(index)

        # Get the augmented sample
        augmented_sample = augmentor.get_augmented_sample(index)

        # Run model inference on augmented sample
        model_output = augmentor.run_model_on_augmented_sample(index)

        # Compute embedding similarity
        es_result = embedding_similarity.compute(augmented_sample, model_output)

        # Compute class activation mapping
        cam_result = cam.compute(augmented_sample, model_output)

        # Store results
        test_results.append({
            "index": index,
            "output": model_output,
            "embedding_similarity": es_result,
            "cam_result": cam_result
        })

    # # Optionally, save and summarize results
    # save_test_results(test_results)
    print_summary(test_results)
