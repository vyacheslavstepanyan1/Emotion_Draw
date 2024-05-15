# **Multiclass Classification Trainer**

This script defines a class `MulticlassClassificationTrainer` for training, evaluating, and making inferences with a multiclass classification model using PyTorch and Hugging Face's Transformers library.

***Note:** You can find this script in `/Emotion_Draw/Emotion_Draw/bert_part/model/`.*

------------------

## **Initialization __init__()**

 ```py 
 __init__(self, model_name, num_labels, batch_size, num_epochs, learning_rate, max_length, device, suffix="")
 ```

#### Arguments:
- `model_name` (str): Name of the pre-trained model from Hugging Face's model hub.
- `num_labels` (int): Number of classes in the classification task.
- `batch_size` (int): Batch size for training.
- `num_epochs` (int): Number of epochs for training.
- `learning_rate` (float): Learning rate for optimization.
- `max_length` (int): Maximum length of input sequences.
- `device` (str): Device to use for training ('cpu' or 'cuda').
- `suffix` (str, optional): Suffix to add to model checkpoint and confusion matrix filenames (default: ' ', `empty string`).

------------------

## **Functions**


------------------
### **_initialize_model()**

Initializes the model, tokenizer, optimizer, loss function, and moves the model to the specified device.

 ```py 
 _initialize_model(self)
 ```

------------------
### **_process_data()**

Processes input DataFrame into tensors for input_ids, attention_masks, and labels.

 ```py 
 _process_data(self, df)
 ```

#### Arguments:
- `df` (DataFrame): Input DataFrame containing 'Sentence' and 'Labels_Encoded' columns.

#### Returns:
- `TensorDataset`: Tensor dataset containing processed data.

------------------
### **save_state()**

Saves the model and optimizer states to a checkpoint file.

 ```py 
 save_state(self, epoch, model_state, optimizer_state, path)
 ```

#### Arguments:
- `epoch` (int): Epoch number.
- `model_state` (dict): State dictionary of the model.
- `optimizer_state` (dict): State dictionary of the optimizer.
- `path` (str): Path to save the checkpoint file.

------------------
### **load_state()**

Loads model and optimizer states from a checkpoint file.

 ```py 
 load_state(self, path)
 ```

#### Arguments:
- `path` (str): Path to the checkpoint file.

#### Returns:
- `int`: Epoch number.
- `dict`: Model state dictionary.
- `dict`: Optimizer state dictionary.

------------------
### **train()**

Trains the model using the provided training and validation datasets.

 ```py 
 train(self, train_df, val_df, log_dir, checkpoint_path=None)
 ```

#### Arguments:
- `train_df` (DataFrame): Training dataset DataFrame.
- `val_df` (DataFrame): Validation dataset DataFrame.
- `log_dir` (str): Directory path to save TensorBoard logs.
- `checkpoint_path` (str, optional): Path to a checkpoint file to resume training (default: None).

------------------
### **save_confusion_matrix()**

Saves the confusion matrix to a file.

 ```py 
 save_confusion_matrix(self, cm, epoch, mode)
 ```

#### Arguments:
- `cm` (array): Confusion matrix array.
- `epoch` (int): Epoch number.
- `mode` (str): Mode of confusion matrix ('train', 'val', 'test').

------------------
### **save_confusion_matrices()**

***Note: Soon to be deprecated.***

Saves training, validation, and test confusion matrices to files.

 ```py 
 save_confusion_matrices(self)
 ```

------------------
### **load_model()**

Loads the model from a checkpoint file.

 ```py 
 load_model(self, checkpoint_path)
 ```

#### Arguments:
- `checkpoint_path` (str): Path to the checkpoint file.

------------------
### **evaluate()**

Evaluates the model on the test dataset.

 ```py 
 evaluate(self, test_df, mode='train')
 ```

#### Arguments:
- `test_df` (DataFrame): Test dataset DataFrame.
- `mode` (str, optional): Evaluation mode ('train' or 'test') (default: 'train').

#### Returns:
- `float`: Average loss.
- `float`: Accuracy.
- `float`: F1 score.
- `array`: Confusion matrix.

------------------
### **single_inference()**

Performs single inference on a given sentence.

 ```py 
 single_inference(self, checkpoint_path, train_df, sentence)
 ```

#### Arguments:
- `checkpoint_path` (str): Path to the checkpoint file.
- `train_df` (DataFrame): DataFrame containing label mappings.
- `sentence` (str): Input sentence for inference.

#### Returns:
- `dict`: Dictionary containing predicted labels and probabilities.

------------------
### **print_cm()**

Prints the confusion matrix.

 ```py 
 print_cm(self, mode, df)
 ```

#### Arguments:
- `mode` (str): Mode of confusion matrix ('train', 'val', 'test').
- `df` (DataFrame): DataFrame containing label mappings.

------------------
