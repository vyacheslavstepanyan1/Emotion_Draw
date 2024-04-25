import torch
import pandas as pd
import os
from ..model.Multiclass_BERT_RoBERTa import MulticlassClassificationTrainer


# Model
model_names = ['bert-base-uncased', 'roberta-base', 'albert-base-v2']
MODEL_NAME = 'albert-base-v2'
SUFFIX = 'experiment1'

#Parameters
num_labels = 6
batch_size = 32
num_epochs = 20
learning_rate = 1e-5
max_length = 128

#Device 
device = torch.device("cpu")

#Initialize
trainer = MulticlassClassificationTrainer(MODEL_NAME, num_labels, batch_size, num_epochs, learning_rate, max_length, device, suffix = SUFFIX)

# Training Dataset
current_dir = os.getcwd()
train_df_dir = os.path.join(current_dir, 'Emotion_Draw/bert_part/data/processed/train_data.csv')
train_df = pd.read_csv(train_df_dir)

def Inference(checkpoint, sentence):
    return trainer.single_inference(checkpoint, train_df, sentence)