# ANLP - SRL
<img src="https://github.com/DavidC001/ANLP/assets/40665241/6228ccbb-da4c-4662-84ce-6216dc5cf5b8" align="left" width="20%" style="margin-right: 15px; margin-bottom: 10px;" />

This project is part of the master course "Applied Natural Language Processing" by Davide Cavicchini. The goal is to develop an SRL system using a transformer encoder model. This repository includes data preprocessing, model training, and inference functionalities.

<br clear="left"/>


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
├── evaluation.py               - Computes the metrics of the saved models on the test set and saves the to a file
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

    Note: to use cuda you might need to install the version of torch that supports your cuda version.

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
<img src="https://github.com/DavidC001/ANLP/assets/40665241/acd4fb1f-cd4a-4dac-a13c-1b9c51c66cd7" align="left" width="40%" style="margin-right: 15px; margin-bottom: 10px;" />

The core model is defined in `model.py`, including the `SRL_MODEL` model class which extends a pre-trained transformer model based on BERT architecture.
For more details on the model architecture, refer to the project report.

<br clear="left"/>

## Training the Model

The training script is managed by `train/main.py`. The datasets need to be properly preprocessed and available in the `datasets/preprocessed` directory.
In the script it's possible to define the tests to run, the model hyperparameters, here is an example of the hyperparameters:

```python
tests = {
        "SRL_DISTILBERT": {
            "model_name": "distilbert-base-uncased", # name of the encoder model to use
            "combine_method": "gating", # how to combine the predicate and word representations, can be "sum", "concat", "soft_attention", "gating"
            "role_layers": [256], # hidden dimensions of the role classifier
            "norm_layer": True, # whether to apply layer normalization
            "proj_dim": 512, # dimension of the projection layer
            "relation_proj": True, # whether to project the relation representation
            "role_RNN": True, # whether to use a LSTM layer in the role classifier
            "RNN_type": "GRU", # type of RNN layer "RNN", "LSTM", "GRU"
            "train_encoder": True, # whether to train the encoder
            "train_embedding_layer": True, # whether to train the embedding layer
            "dropout_prob": 0.5, # dropout rate
        },
}
```

For more information on how these modify the model refer to the documentation in the `model.py` file.
## Evaluation Metrics

The metrics used are Precision, Recall, and F1-score. These metrics are computed for both the identification of prepositions and the classification of semantic roles in the sentences.

To compute these metrics for the models in the `models` directory you can ran the script `train/evaluation.py` which will evaluate all the models in the directory and save the results to a json file. When calling the script you can choose what argument span identification strategy to evaluate using the `--top` and `--concat` flags. You can also control the confidence threshold for the span identification using the `--threshold value` flag.

For example:
    ```bash
    python ./train/evaluation.py --top --threshold 0.75
    ```

## Inference

There are two scripts for inference:
- `inference.py`: This script allows you to make predictions on new sentences from the terminal, to use it you need to adjust the paths and have a trained model available with its configuration json file.
- `interactive_KG.py`: This script generates an interactive Knowledge Graph using as input a file (the default file is `text.txt` from the root directory of the repository)  or a wikipedia page. To use it you need to set up an instance of [Neo4j](https://neo4j.com/) and adjust the credentials in the script.

## Trained Models

The checkpoints and configuration files for the trained models are available in the following drive [folder](https://drive.google.com/drive/folders/18aqSxo3HIs4e1XQCdYkuUqP5_P8cyNVd?usp=sharing).
The evaluation metrics for the models are available in the following drive [folder](https://drive.google.com/drive/folders/18pG4_AUdEhkgXMq4ZbyripT7SpwgVXuO?usp=sharing)

## License

This project is licensed under the MIT License.
