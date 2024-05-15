# **Step-by-Step: Fine-Tuning Bert and Friends** 👣

In this section, you'll find the complete end-to-end process for fine-tuning BERT-based sequence classification models on an emotion dataset created for natural language processing (NLP) tasks. The goal is to train models capable of accurately classifying emotions expressed in text data.

***Note:** You can find an associated notebook in `/Emotion_Draw/Emotion_Draw/bert_part/notebooks/BERT-based_Sequence_Classification.ipynb`.*

## **Import Packages**

We begin by importing necessary packages for data manipulation, model training, and evaluation. 

```python
import torch
import pandas as pd
import os
import logging

import sys
sys.path.append(os.path.dirname(os.getcwd()))

from model.Multiclass_BERT import MulticlassClassificationTrainer
logging.getLogger("matplotlib.colorbar").setLevel(logging.ERROR)
```

## **Choose a Model**

Select a BERT-based model to train for the classification task (feel free to add other similar models). Set a suffix to distinguish different experiments.

```python
model_names = ['bert-base-uncased', 'roberta-base', 'albert-base-v2']
MODEL_NAME = 'albert-base-v2'
SUFFIX = 'experiment-tech-test'
```

## **Specify the Parameters**

Define the parameters such as the number of labels, batch size, number of epochs, learning rate, maximum sequence length, and device for training.

```python
num_labels = 6
batch_size = 32
num_epochs = 3
learning_rate = 1e-5
max_length = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## **Initialize the Class**

Instantiate the `MulticlassClassificationTrainer` class with the chosen model and specified parameters.

```python
trainer = MulticlassClassificationTrainer(MODEL_NAME, num_labels, batch_size, num_epochs, learning_rate, max_length, device, suffix=SUFFIX)
```

## **Load Data**

Load the training and validation datasets from CSV files.

```python
train_df = pd.read_csv('../data/processed/train_data.csv')
val_df = pd.read_csv('../data/processed/val_data.csv')
train_df = train_df.head(200)  # Limiting data for a quick test
val_df = val_df.head(200)  # Limiting data for a quick test
```

## **Training**

Train the model using the training dataset and validate it using the validation dataset. TensorBoard logs are written to the specified directory for visualization.

```python
log_dir = f'../runs/{MODEL_NAME}_{SUFFIX}'
trainer.train(train_df, val_df, log_dir)
```

## **Evaluate on the Test Set**

Load the best-performing model checkpoint and evaluate it on the test dataset.

```python
model_path = "../models_trained/multiclass_experiment-tech-test_albert-base-v2_best_checkpoint.pth"
trainer.load_model(model_path)
test_df = pd.read_csv('../data/processed/test_data.csv')
test_df = test_df.head(100)  # Limiting data for a quick test
trainer.evaluate(test_df, mode='test')
```

## **Inference for a Single Example**

Perform inference for individual sentences using the trained model.

```python
checkpoint = "../models_trained/multiclass_experiment-tech-test_albert-base-v2_best_checkpoint.pth"
sentence = "My boss made me do all the frustrating work."
trainer.single_inference(checkpoint, train_df, sentence)
```

## **Display Confusion Matrices**

Visualize the confusion matrices for the training, validation, and test sets.

```python
trainer.print_cm(mode="train", df=train_df)
trainer.print_cm(mode="val", df=val_df)
trainer.print_cm(mode="test", df=test_df)
```

## **TensorBoard**

Use TensorBoard to visualize the training process. Start TensorBoard on port `6008` (adjust the port as needed) and monitor the training logs.

```bash
%reload_ext tensorboard
%tensorboard --logdir=../runs --port=6008 --load_fast=false
```

If the specified port is busy, identify the PID (Process ID) using `!lsof -i :6008` and terminate it using `!kill PID`. 

