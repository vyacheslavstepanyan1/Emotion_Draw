"""
This script sets up a Multiclass BERT RoBERTa model for emotion classification and provides a function for inference.

The script performs the following tasks:
- Imports necessary libraries and modules.
- Defines model parameters such as model names, suffix, number of labels, batch size, number of epochs, learning rate, maximum sequence length, and device.
- Initializes a MulticlassClassificationTrainer instance with the specified parameters.
- Loads the training dataset from a CSV file.
- Defines a function for performing inference using the trained model checkpoint and a given input sentence.

Usage:
- Set the desired model name, suffix, and other parameters.
- Initialize the MulticlassClassificationTrainer instance.
- Load the training dataset from the CSV file.
- Use the provided function "Inference" to perform inference on input sentences using the trained model checkpoint.
"""

# Import necessary libraries 
import torch
import pandas as pd
import os
from ..model.Multiclass_BERT_RoBERTa import MulticlassClassificationTrainer

# Define model parameters
model_names = ['bert-base-uncased', 'roberta-base', 'albert-base-v2']
MODEL_NAME = model_names[2] # Change depending on the checkpoint model.

# These parameters are used for initializing the class instance but are not directly involved in the process of making inference.
num_labels = 6 
batch_size = 32
num_epochs = 20
learning_rate = 1e-5
max_length = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MulticlassClassificationTrainer
trainer = MulticlassClassificationTrainer(MODEL_NAME, num_labels, batch_size, num_epochs, learning_rate, max_length, device)

# Load training dataset
current_dir = os.getcwd()
train_df_dir = os.path.join(current_dir, 'Emotion_Draw/bert_part/data/processed/train_data.csv')
train_df = pd.read_csv(train_df_dir)

def Inference(checkpoint, sentence):
    """
    Perform inference on a given sentence using the trained model.

    Args:
    - checkpoint (str): Path to the checkpoint file.
    - sentence (str): Input sentence for inference.

    Returns:
    - dict: Dictionary containing predicted labels and probabilities.
    """
    return trainer.single_inference(checkpoint, train_df, sentence)
