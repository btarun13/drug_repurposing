
# import mlflow
import torch

from src import (
    LightGCN,
    prepare_data,
    save_model_pickle,
    train_and_evaluate,
)


def run_pipeline(
    ground_truth_path : str ,
    node_embeddings_path : str,
    save_path,
    epochs :int ,
    batch_size:int,
    num_layers:int,
    test_size:int ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = prepare_data(ground_truth, node_embeddings, test_size= test_size) ###  80/20 split

    model = LightGCN(disease_embeddings=data['disease_embeddings'],
                     drug_embeddings=data['drug_embeddings'],
                     num_layers = num_layers).to(device)
    
        # Start MLflow run
    # with mlflow.start_run():
    #     # Log hyperparameters
    #     mlflow.log_param("epochs", epochs)
    #     mlflow.log_param("batch_size", batch_size)
    #     mlflow.log_param("num_layers", num_layers)
    #     mlflow.log_param("test_size", test_size)

    #     # Train and evaluate
        train_and_evaluate(model, data, num_epochs=epochs, batch_size=batch_size, device=device)

    #     # Save model
        save_model_pickle(model, data, save_path)

    #     # Optionally, log the model to MLflow
    #     mlflow.pytorch.log_model(model, artifact_path="models")



if __name__ == '__main__':

     # Example usage
    main(
        ground_truth_path="/Users/tarun/0_REPOS/drug_repurposing/Ground_Truth.csv",
        node_embeddings_path="/Users/tarun/0_REPOS/drug_repurposing/node_embeddings.csv",
        save_path="/Users/tarun/0_REPOS/drug_repurposing/saved_model.pkl",
        epochs=60,
        batch_size=100,
        num_layers=4,
        test_size=0.2
    )
    # save_path = '/content/saved_model.pkl' # Path to save the model
    # main(save_path)
