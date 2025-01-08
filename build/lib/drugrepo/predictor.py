

import pandas as pd
import torch


def get_pair_score(model: torch.nn.Module,
                   drug_id: str, 
                   disease_id: str, 
                   data: dict, 
                   device: str) -> dict:
    
    """
    Get interaction score for a specific drug-disease pair

    Args:
        model: Trained LightGCN model
        drug_id: Drug identifier (e.g., 'CHEMBL123456')
        disease_id: Disease identifier (e.g., 'MONDO:0007186')
        data: Dictionary containing model data and mappings
        device: torch device

    Returns:
        dict: Dictionary containing the score and relevant information
    """
    model.eval()

    # Check if drug and disease exist in our mappings
    if drug_id not in data['drug_to_idx']:
        raise ValueError(f"Drug ID {drug_id} not found in training data")
    if disease_id not in data['disease_to_idx']:
        raise ValueError(f"Disease ID {disease_id} not found in training data")

    # Get indices
    drug_idx = data['drug_to_idx'][drug_id]
    disease_idx = data['disease_to_idx'][disease_id]

    with torch.no_grad():
        # Get embeddings
        disease_emb_final, drug_emb_final = model(data['train_edge_index'].to(device))

        # Get specific embeddings
        disease_emb = disease_emb_final[disease_idx].unsqueeze(0)
        drug_emb = drug_emb_final[drug_idx].unsqueeze(0)

        # Calculate score
        score = torch.mm(disease_emb, drug_emb.t()).squeeze()
        probability = torch.sigmoid(score)

        return {
            'drug_id': drug_id,
            'disease_id': disease_id,
            'raw_score': score.item(),
            'probability': probability.item(),
            'disease_emb':list(disease_emb),
            'drug_emb':list(drug_emb)
        }

def generate_recommendations_for_disease(model: torch.nn.Module, 
                                         disease_id: str, 
                                         ground_truth: pd.DataFrame, 
                                         data: dict, 
                                         device: str, 
                                         top_k: int) -> pd.DataFrame:
    """
    Generate ranked drug recommendations for a specific disease

    Args:
        model: Trained LightGCN model
        disease_id: Disease identifier (e.g., 'MONDO:0007186')
        ground_truth: Original ground truth DataFrame
        data: Dictionary containing model data and mappings
        device: torch device
        top_k: Number of recommendations to return

    Returns:
        DataFrame with ranked drug recommendations and their scores
    """
    model.eval()

    # Get disease index
    disease_idx = data['disease_to_idx'][disease_id]

    # Get existing positive interactions
    existing_interactions = set(
        ground_truth[
            (ground_truth['target'] == disease_id) &
            (ground_truth['y'] == 1)
        ]['source'].values
    )

    with torch.no_grad():
        # Get embeddings
        disease_emb_final, drug_emb_final = model(data['train_edge_index'].to(device))

        # Get disease embedding
        disease_emb = disease_emb_final[disease_idx].unsqueeze(0)

        # Calculate scores for all drugs
        scores = torch.mm(disease_emb, drug_emb_final.t()).squeeze()
        # scores = torch.sigmoid(scores)  # Convert to probabilities or raw scores

        # Convert to numpy for easier handling
        scores = scores.cpu().numpy()

        # Create recommendations DataFrame
        recommendations = []
        for drug_id, drug_idx in data['drug_to_idx'].items():
            if drug_id not in existing_interactions:  # Exclude existing positive interactions
                recommendations.append({
                    'drug_id': drug_id,
                    'score': scores[drug_idx],
                })

        recommendations_df = pd.DataFrame(recommendations)

        # Sort by score and get top k
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(top_k)
        recommendations_df = recommendations_df.reset_index(drop=True)

        # Add rank column
        recommendations_df['rank'] = recommendations_df.index + 1

        # Reorder columns
        recommendations_df = recommendations_df[['rank', 'drug_id', 'score']]

    return recommendations_df
