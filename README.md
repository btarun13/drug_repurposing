# Drug Repurposing (Knowledge Embedding and Classifier)

This project consists of two Colab notebooks that require GPU support for execution. All necessary data files are included in this repository.

## Question 1: Node Embedding Generation
For the initial node embeddings, I extracted the 'all_names', 'description', and 'label' columns from the Nodes.csv file. Using the 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' transformer, I generated 200-dimensional embeddings for each node based on this textual information.
The second phase involved learning and updating these embeddings using knowledge graph connections from Edges.csv. Due to time constraints, I've included comments describing my intended approach. While I explored the RTXteam repository, integrating their codebase would have required more time. Instead, I implemented solutions using familiar methodologies for faster development.

## Question 2: Link Classification
I utilized Ground Truth.csv and recommendation_pipeline_initial_node_embeddings.csv (a preprocessed dataset) for this task. Recognizing the problem's bipartite nature, I implemented LightGCN with binary labels (1/0).
The preprocessed file was created from Nodes.csv by:

1. Extracting a subset of nodes present in Ground Truth.csv
2. Processing topological embedding columns
3. Applying string formatting

The resulting file "recommendation_pipeline_initial_node_embeddings.csv" contains ID, name, and their 128-dimensional embeddings.

## Example code:
```python

ground_truth = pd.read_csv('/content/Ground Truth.csv')
node_embeddings = pd.read_csv('/content/recomendation_pipeline_initial_node_embeddings.csv')

```
These datasets are used in the second notebook with detailed instructions.
All required libraries are available in Google Colab with A100 or T4 GPUs. Install prerequisites using:

```python

!pip install torch_geometric

```
## Features

Generation of initial embeddings using transformers based on textual information, followed by embedding refinement using knowledge graph connections (Q1)
Data preprocessing and LightGCN implementation for drug-disease link classification based on learned embeddings, with sigmoid activation
Additional functionality for drug recommendations based on specific diseases
Standardized terminology: "smallMolecule" and "drugs" categories were consolidated as "drugs", while the target category was designated as "diseases" to facilitate bipartite graph analysis

## Results
All implementation steps are documented in the notebooks. Final performance metrics:
-Epoch 60:
-Loss: 0.1406
-Test AUC-ROC: 0.9494
-Test AUC-PR: 0.9233



## Acknowledgments
Thank you for the opportunity to work on this test project.


