
import torch.nn.functional as F
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
import torch


def train_epoch(model: torch.nn.Module, 
                train_edge_index: torch.Tensor, 
                train_pairs: torch.Tensor, 
                train_labels: torch.Tensor, 
                optimizer, 
                batch_size: int, 
                device: str = "cuda"):
    
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = (len(train_pairs) + batch_size - 1) // batch_size

    # Shuffle training data
    indices = torch.randperm(len(train_pairs))
    train_pairs = train_pairs[indices]
    train_labels = train_labels[indices]

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(train_pairs))

        batch_pairs = train_pairs[start_idx:end_idx].to(device)
        batch_labels = train_labels[start_idx:end_idx].to(device)

        optimizer.zero_grad()

        # Get predictions
        predictions = model.predict(
            batch_pairs[:, 0],
            batch_pairs[:, 1],
            train_edge_index.to(device)
        )

        # Binary cross entropy loss
        loss = F.binary_cross_entropy(predictions, batch_labels)

        # Add L2 regularization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        loss += 0.0001 * l2_reg

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches

def evaluate(model: torch.nn.Module, 
             train_edge_index: torch.Tensor, 
             test_pairs: torch.Tensor, 
             test_labels: torch.Tensor, 
             device: str = "cuda") -> dict:
    
    """Evaluate the model"""
    model.eval()
    with torch.no_grad():
        test_predictions = model.predict(
            test_pairs[:, 0].to(device),
            test_pairs[:, 1].to(device),
            train_edge_index.to(device)
        )

        # Calculate metrics
        auc_roc = roc_auc_score(test_labels.cpu(), test_predictions.cpu())
        precision, recall, _ = precision_recall_curve(test_labels.cpu(), test_predictions.cpu())
        auc_pr = auc(recall, precision)

        return {
            'AUC-ROC': auc_roc,
            'AUC-PR': auc_pr
        }

def train_and_evaluate(model: torch.nn.Module, 
                       data : dict, 
                       num_epochs : int, 
                       batch_size : int, 
                       lr : float, 
                       device='cuda'):
    
    
    """Train and evaluate the model"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(
            model,
            data['train_edge_index'],
            data['train_pairs'],
            data['train_labels'],
            optimizer,
            batch_size,
            device
        )

        # Evaluate
        if (epoch + 1) % 10 == 0:
            metrics = evaluate(
                model,
                data['train_edge_index'],
                data['test_pairs'],
                data['test_labels'],
                device
            )
            print(f'Epoch {epoch+1}: Loss = {train_loss:.4f}, '
                  f'Test AUC-ROC = {metrics["AUC-ROC"]:.4f}, '
                  f'Test AUC-PR = {metrics["AUC-PR"]:.4f}')
            
    return model
            # Log metric to MLflow
        # mlflow.log_metric("loss", loss.item(), step=epoch)

