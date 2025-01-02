import pickle
import torch
from drugrepo.model import LightGCN

def save_model_pickle(model, data, save_path: str):
    # Move model to CPU for saving
    model = model.cpu()

    # Create save dictionary
    save_dict = {
        'model_state': model.state_dict(),
        'model_config': {
            'num_diseases': model.num_diseases,
            'num_drugs': model.num_drugs,
            'embedding_dim': model.embedding_dim,
            'num_layers': model.num_layers
        },
        'mappings': {
            'drug_to_idx': data['drug_to_idx'],
            'disease_to_idx': data['disease_to_idx']
        },
        'train_edge_index': data['train_edge_index'].cpu()
    }

    # Save using pickle
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"Model saved to {save_path}")

def load_model_pickle(load_path, device='cuda'):
    # Load the pickle file
    with open(load_path, 'rb') as f:
        save_dict = pickle.load(f)

    # Initialize model with saved configuration
    model = LightGCN(
        disease_embeddings=torch.randn(save_dict['model_config']['num_diseases'],
                                     save_dict['model_config']['embedding_dim']),
        drug_embeddings=torch.randn(save_dict['model_config']['num_drugs'],
                                  save_dict['model_config']['embedding_dim']),
        num_layers=save_dict['model_config']['num_layers']
    )

    # Load model state
    model.load_state_dict(save_dict['model_state'])

    # Move model to specified device
    model = model.to(device)

    # Prepare data dictionary
    data = {
        'drug_to_idx': save_dict['mappings']['drug_to_idx'],
        'disease_to_idx': save_dict['mappings']['disease_to_idx'],
        'train_edge_index': save_dict['train_edge_index'].to(device)
    }

    return model, data
