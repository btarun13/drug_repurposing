# Drug Repurposing (Recommendation package)

Experimental Package for building models and stacking them for recommendation score. We can also integrate them in MLflow for logging parameters.

## To install and use package from github repo

```
!pip install git+https://github.com/btarun13/drug_repurposing.git@feature_branch

```

In case you are using VScode, .toml files will have everything. If you have uv package manager, just run

```bash

uv sync

```

With this, you would be in the same enviroment setting used for the package.

## Example code:
```python

from drugrepo.data_loader import prepare_data
from drugrepo.model import LightGCN
from drugrepo.model_train import train_and_evaluate
import torch
from drugrepo.predictor import get_pair_score

```

At the moment there is only one type of architecture specified at in drugrepo.model . We can have more architectures defined within and use for training models with different architecture. For now, we will continue with LightGCN

```python

input_data = prepare_data("/content/Ground Truth.csv",        #### path for edge data
             "/content/recomendation_pipeline_initial_node_embeddings.csv",  ### oath for node data
             test_size=0.3,   ### test size and random states can vary, in cause you have a hold out validation set we can use multiple train/test spilts with a lot of different seeds to measure performance on validation set
             random_state= 123)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```


## Training with specific hyperparameters

```python
layer = 2
model = LightGCN(
    disease_embeddings=input_data['disease_embeddings'],
    drug_embeddings=input_data['drug_embeddings'],
    num_layers=layer).to(device)

train_and_evaluate(model, input_data, num_epochs=20, batch_size=50, lr = 0.01, device=device)  ### change hyperparameters

```
```bash
Epoch 10: Loss = 0.0556, Test AUC-ROC = 0.9604, Test AUC-PR = 0.9348
Epoch 20: Loss = 0.0564, Test AUC-ROC = 0.9617, Test AUC-PR = 0.9330

```



## Results


```python
get_pair_score(model,
               'CHEMBL.COMPOUND:CHEMBL30',
               'MONDO:0007362', 
               input_data,
               device)

```

```bash
{'drug_id': 'CHEMBL.COMPOUND:CHEMBL30',
 'disease_id': 'MONDO:0007362',
 'raw_score': 1.5422860383987427,
 'probability': 0.8237967491149902,
 'disease_emb': [tensor([-5.9911e-02, -1.2041e-01,  5.0482e-02, -3.2367e-03,  9.9549e-02,
           7.9791e-02, -2.2457e-01,  3.2794e-01,  1.5935e-01, -1.9004e-01,
          -3.4828e-03, -1.3589e-01,  2.1126e-01,  5.3988e-01,  1.8098e-01,
           2.1025e-02, -6.7591e-02,  1.2305e-01,  6.2050e-02, -2.1232e-01,
           4.6071e-02,  1.8825e-01, -1.5292e-01,  3.0699e-01, -1.1182e-01,
           6.2287e-02,  1.0639e-01,  3.0332e-01, -2.0131e-01,  8.1552e-02,
           3.6927e-01,  1.5867e-01,  1.3609e-01,  4.8282e-02, -3.2262e-02,
          -6.1631e-02, -4.8148e-02,  1.4393e-01,  7.9521e-02, -1.3832e-01,
           2.2115e-01,  3.3398e-02,  2.3921e-01,  1.4900e-01, -1.3909e-01,
           1.5112e-01, -3.5351e-01, -7.8104e-02, -1.2823e-02,  1.3334e-01,
          -7.3039e-02, -2.5402e-01,  1.5269e-01,  2.0441e-01,  4.7075e-02,
          -1.9494e-01,  1.1327e-01, -2.3754e-01, -1.5045e-01, -2.8113e-01,
           2.0229e-01, -2.3288e-01, -6.3663e-02,  5.2580e-02,  4.5360e-02,
          -2.4321e-02,  3.8664e-01,  1.1374e-01, -2.9992e-02, -4.5821e-02,
          -1.2341e-01,  3.0121e-04, -3.1351e-01, -2.1253e-02,  7.1962e-02,
          -1.3010e-01,  2.4236e-01,  3.7549e-02,  1.1319e-01, -2.4052e-01,
           4.5178e-02,  4.5833e-02, -7.6962e-02, -8.9253e-02,  3.1187e-01,
          -1.9854e-01,  1.3689e-01, -2.7576e-01,  1.1196e-01,  5.2103e-02,
          -1.6629e-01,  9.4572e-02,  6.4623e-03, -3.1499e-01, -6.8299e-03,
          -1.4604e-01, -2.0032e-01, -1.9492e-01, -2.8079e-03, -4.1250e-02,
          -4.3154e-01, -1.1046e-01,  2.9363e-01,  1.6861e-02, -1.5691e-01,
          -4.0218e-02,  1.5355e-01,  3.2216e-01,  1.2064e-01,  3.6648e-02,
           9.5431e-02,  1.7660e-01, -6.4549e-02, -1.4671e-01, -4.7712e-02,
          -1.4081e-02,  4.6027e-02,  9.9893e-02, -4.8742e-03,  2.2973e-01,
           5.4576e-01, -2.1284e-01,  5.7996e-02,  1.0800e-01, -3.6115e-02,
          -1.6547e-01, -2.2726e-01, -5.1840e-02], device='cuda:0')],
 'drug_emb': [tensor([-2.6009e-01,  1.3955e-01,  3.4041e-01, -2.9036e-01,  2.3295e-01,
           1.8259e-01,  4.3216e-01,  1.7057e-01,  4.2607e-01,  4.9822e-01,
           2.3535e-01, -2.9875e-01,  2.2269e-01, -3.2226e-01,  4.0209e-01,
          -2.2594e-01, -8.0527e-02, -1.2607e-01, -2.5287e-01, -1.4696e-01,
           3.8881e-01,  1.5083e-01, -9.1018e-02,  1.9653e-01,  3.4487e-01,
           2.9561e-01,  5.7577e-02,  1.6204e-01, -1.7187e-01,  2.3540e-01,
           1.9444e-01,  1.9209e-01, -2.6476e-01,  3.4033e-01,  1.5154e-01,
          -2.7733e-01,  1.5178e-01, -2.1976e-01, -3.6034e-01, -7.7930e-02,
           1.5717e-01,  2.1479e-01,  3.2736e-01,  7.0454e-02, -7.2268e-02,
          -5.2160e-01, -1.8327e-01,  2.9841e-02, -3.4221e-01, -1.2182e-01,
           3.6185e-01,  6.9399e-02,  1.6537e-01, -6.5987e-02,  3.4278e-01,
          -2.4307e-01,  6.2978e-02, -3.2228e-01, -2.1620e-01,  1.1588e-02,
          -1.3403e-01,  4.1289e-02,  7.8002e-02, -4.5184e-01,  3.4374e-01,
           4.3088e-01,  9.7138e-01,  1.2647e-01, -3.5435e-01, -3.4411e-01,
          -9.2454e-04,  2.2160e-01, -5.5603e-01,  4.0592e-02, -4.9680e-02,
          -8.2038e-02,  2.5924e-01, -3.8464e-01,  6.3594e-02, -3.4363e-01,
           3.4359e-01,  3.4412e-01,  3.4924e-01, -1.4866e-01,  2.0214e-01,
          -1.6499e-01,  1.7197e-01, -1.5671e-01,  4.3623e-02,  3.2187e-01,
          -1.3201e-01,  3.8828e-02, -1.3456e-01,  7.3604e-01,  6.0945e-02,
          -2.6232e-01,  1.3944e-01, -1.4865e-01,  3.6801e-01, -8.6408e-02,
          -7.9694e-01, -1.3776e-01,  6.4576e-01, -4.2333e-02, -1.3449e-01,
           1.5805e-01,  1.6498e-01, -3.2644e-03,  8.6759e-02, -2.8934e-01,
           4.1675e-01,  3.7614e-01, -1.1607e-01,  2.0894e-01, -1.5386e-01,
          -2.0994e-01,  3.4077e-01,  3.0947e-02, -2.1975e-01, -1.1292e-01,
          -4.9965e-01,  1.2497e-01, -2.1930e-01,  1.5485e-02,  2.6300e-01,
          -2.5049e-02, -7.2167e-02,  4.0921e-01], device='cuda:0')]}

```





## Acknowledgments
Thank you for the opportunity to work on this test project.


