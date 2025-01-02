

import torch.nn as nn
import torch

class LightGCN(nn.Module):
    def __init__(self, disease_embeddings, drug_embeddings, num_layers):
        super(LightGCN, self).__init__()

        # Convert numpy arrays to torch tensors if they aren't already
        if not isinstance(disease_embeddings, torch.Tensor):
            disease_embeddings = torch.FloatTensor(disease_embeddings)
        if not isinstance(drug_embeddings, torch.Tensor):
            drug_embeddings = torch.FloatTensor(drug_embeddings)

        # Create embedding layers and register as parameters
        self.num_diseases = disease_embeddings.shape[0]
        self.num_drugs = drug_embeddings.shape[0]
        self.embedding_dim = disease_embeddings.shape[1]

        # Register embeddings as parameters
        self.disease_embedding = nn.Parameter(disease_embeddings)
        self.drug_embedding = nn.Parameter(drug_embeddings)
        self.num_layers = num_layers

    def forward(self, edge_index):
        # Get embeddings
        diseases_emb = self.disease_embedding
        drugs_emb = self.drug_embedding
        all_emb = torch.cat([diseases_emb, drugs_emb])

        # Storage for embeddings at each layer
        embs = [all_emb]

        # Compute adjacency matrix
        adj = torch.zeros((all_emb.shape[0], all_emb.shape[0]), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1

        # Compute degree matrix
        degree = adj.sum(dim=1)
        degree_sqrt = torch.sqrt(degree + 1e-12)
        degree_matrix_inv_sqrt = torch.diag(1.0 / degree_sqrt)

        # Normalize adjacency matrix
        norm_adj = degree_matrix_inv_sqrt @ adj @ degree_matrix_inv_sqrt

        # Message passing layers
        for _ in range(self.num_layers):
            all_emb = norm_adj @ all_emb
            embs.append(all_emb)

        # Final embeddings are mean of all layers
        final_embs = torch.stack(embs, dim=0).mean(dim=0)

        diseases_emb_final, drugs_emb_final = torch.split(final_embs, [self.num_diseases, self.num_drugs])
        return diseases_emb_final, drugs_emb_final

    def predict(self, disease_indices, drug_indices, edge_index):
        diseases_emb_final, drugs_emb_final = self.forward(edge_index)
        disease_emb = diseases_emb_final[disease_indices]
        drug_emb = drugs_emb_final[drug_indices]

        # Compute prediction scores
        predictions = (disease_emb * drug_emb).sum(dim=1)
        return torch.sigmoid(predictions)
