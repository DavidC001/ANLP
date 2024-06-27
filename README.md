# ANLP

This repository contains code and scripts for performing Semantic Role Labeling (SRL) using BERT architecture. It includes data preprocessing, model training, and evaluation functionalities.

## Directory Structure

```
├── ConceptNET                  - Unused scripts to generate data from ConceptNET
|   ├── ...
|
├── dataloaders
│   ├── pre-process.py          - General preprocessing script
│   ├── preprocess_nombank.py   - Preprocessing script for NomBank dataset
│   ├── preprocess_UP.py        - Preprocessing script for Universal Proposition dataset
│   ├── UP_dataloader.py        - Data loader for Universal Proposition dataset
│   ├── NOM_dataloader.py       - Data loader for NomBank dataset
|
├── inference                  
│   ├── inference.py            - Inference script for making predictions on new sentences from terminal
│   ├── interactive_KG.py       - Generates an interactive Knowledge Graph using as input a file or a wikipedia page
|
├── train
│   ├── functions.py            - Functions for training the model and evaluating the results
│   ├── utils.py                - Utility functions to load the data
│   ├── main.py                 - Main script for training the model
|
├── model.py                    - Core model architecture
├── requirements.txt            - Python dependencies
|
├── .gitignore                  - Git ignore file
├── README.md                   - Project README file
```

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/srl-bert.git
    cd srl-bert
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Data pre-preprocessing

Before training the model, you need to preprocess the data. The `dataloaders` directory contains scripts for preprocessing different datasets:

- `preprocess_UP.py`: To preprocess the Universal Proposition dataset.
- `preprocess_nombank.py`: To preprocess the NomBank dataset. Modified from this [implementation](https://github.com/CogComp/SRL-English.git)
- `pre-process.py`: To preprocess both datasets.

To run these scripts, you need to adjust the paths and parameters according to your setup.
These are the links to download the datasets:
- [Universal Propositions](https://github.com/UniversalPropositions/UP-1.0.git)
- [NomBank](https://nlp.cs.nyu.edu/meyers/NomBank.html)

For the paths to adjust in the script to preprocess the nomBank dataset make reference to this [repository](https://github.com/CogComp/SRL-English.git). Note, you also need to adjust the paths and download [ALGNLP_2](https://github.com/Hai-Pham/ALGNLP_2.git) data.

## Model Architecture

The core model is defined in `model.py`, which includes the `SRL_MODEL` which extends a pre-trained transformer model based on BERT architecture.
For more details on the model architecture, refer to the project report.

## Training the Model

The training script is managed by `train/main.py`. The datasets need to be properly preprocessed and available in the `datasets/preprocessed` directory.
In the script it's possible to define the tests to run, the model hyperparameters, here is an example of the hyperparameters:

```python
tests = {
        "SRL_DISTILBERT_gated_redboth100_100_norm_cosineLR_weightedLoss": {
            "model_name": "distilbert/distilbert-base-uncased",
            "combine_method": "gating",
            "role_layers": [100],
            "role_LSTM": False,
            "norm_layer": True,
            "dim_reduction": 100,
            "relation_reduction": True,
            "train_encoder": True
        },
}
```

For more information on how these modify the model refer to the documentation in the `model.py` file.

## Evaluation Metrics

The metrics used are Precision, Recall, and F1-score. These metrics are computed for both the identification of relations and classification of semantic roles in the sentences.

## Inference

There are two scripts for inference:
- `inference.py`: This script allows you to make predictions on new sentences from the terminal, to use it you need to adjust the paths and have a trained model available with its configuration json file.
- `interactive_KG.py`: This script generates an interactive Knowledge Graph using as input a file or a wikipedia page. To use it you need to set up an instance of [Neo4j](https://neo4j.com/) and adjust the credentials in the script.

## Trained Models

The checkpoints and configuration files for the trained models are available in the following drive [folder](https://drive.google.com/drive/folders/18aqSxo3HIs4e1XQCdYkuUqP5_P8cyNVd?usp=sharing).
The evaluation metrics for the models are available in the following drive [folder](https://drive.google.com/drive/folders/18pG4_AUdEhkgXMq4ZbyripT7SpwgVXuO?usp=sharing)

## License

This project is licensed under the MIT License.
