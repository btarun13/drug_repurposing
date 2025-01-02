
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def prepare_data(ground_truth_path : str , 
                 node_embeddings_path : str, 
                 test_size : float , 
                 random_state : int) -> dict:
    """
    Prepare train and test datasets using the ground truth data and initial embeddings
    """

    ground_truth = pd.read_csv(ground_truth_path )
    node_embeddings = pd.read_csv(node_embeddings_path)


    # Create mapping dictionaries for diseases and drugs
    disease_to_idx = {disease: idx for idx, disease in enumerate(ground_truth['target'].unique())}
    drug_to_idx = {drug: idx for idx, drug in enumerate(ground_truth['source'].unique())}

    # Extract embeddings from node_embeddings DataFrame
    drug_embeddings = []
    disease_embeddings = []

    # Get the embedding columns
    embedding_cols = [col for col in node_embeddings.columns if col.startswith('embedding_')]

    # Create embeddings matrices
    for drug in drug_to_idx:
        drug_embedding = node_embeddings[node_embeddings['id'] == drug][embedding_cols].values[0]
        drug_embeddings.append(drug_embedding)

    for disease in disease_to_idx:
        disease_embedding = node_embeddings[node_embeddings['id'] == disease][embedding_cols].values[0]
        disease_embeddings.append(disease_embedding)

    drug_embeddings = np.stack(drug_embeddings)
    disease_embeddings = np.stack(disease_embeddings)

    # Split the data into train and test
    train_df, test_df = train_test_split(ground_truth, test_size=test_size,
                                        random_state=random_state, stratify=ground_truth['y'])

    # Create edge indices for training (using only positive edges)
    train_edge_index = []
    for _, row in train_df[train_df['y'] == 1].iterrows():
        drug_idx = drug_to_idx[row['source']]
        disease_idx = disease_to_idx[row['target']]
        train_edge_index.append([drug_idx, disease_idx + len(drug_to_idx)])
        train_edge_index.append([disease_idx + len(drug_to_idx), drug_idx])

    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t()

    # Prepare train pairs and labels (including both positive and negative examples)
    train_pairs = []
    train_labels = []
    for _, row in train_df.iterrows():
        drug_idx = drug_to_idx[row['source']]
        disease_idx = disease_to_idx[row['target']]
        train_pairs.append([disease_idx, drug_idx])
        train_labels.append(row['y'])

    # Prepare test pairs and labels
    test_pairs = []
    test_labels = []
    for _, row in test_df.iterrows():
        drug_idx = drug_to_idx[row['source']]
        disease_idx = disease_to_idx[row['target']]
        test_pairs.append([disease_idx, drug_idx])
        test_labels.append(row['y'])

    return {
        'train_edge_index': train_edge_index,
        'train_pairs': torch.tensor(train_pairs, dtype=torch.long),
        'train_labels': torch.tensor(train_labels, dtype=torch.float),
        'test_pairs': torch.tensor(test_pairs, dtype=torch.long),
        'test_labels': torch.tensor(test_labels, dtype=torch.float),
        'disease_to_idx': disease_to_idx,
        'drug_to_idx': drug_to_idx,
        'disease_embeddings': disease_embeddings,
        'drug_embeddings': drug_embeddings
    }
