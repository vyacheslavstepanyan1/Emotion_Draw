"""
Multiclass Classification Trainer

This script defines a class `MulticlassClassificationTrainer` for training, evaluating, and making inferences
with a multiclass classification model using PyTorch and Hugging Face's Transformers library.

"""

# Importing necessary libraries
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For numerical operations
import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
import pickle  # For serializing and deserializing Python objects
import seaborn as sns  # For data visualization
import shutil  # For high-level file operations
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score  # For evaluating model performance
import torch  # For building deep learning models
from torch.optim import AdamW  # For using the AdamW optimizer
from torch.utils.data import DataLoader, TensorDataset  # For handling data loading in PyTorch
from torch.utils.tensorboard import SummaryWriter  # For writing TensorBoard logs
from tqdm import tqdm  # For displaying progress bars
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer  # For using pre-trained transformer models


class MulticlassClassificationTrainer:
    def __init__(self, model_name, num_labels, batch_size, num_epochs, learning_rate, max_length, device, suffix=""):
        """
        Initializes the MulticlassClassificationTrainer.

        Args:
        - model_name (str): Name of the pre-trained model from Hugging Face's model hub.
        - num_labels (int): Number of classes in the classification task.
        - batch_size (int): Batch size for training.
        - num_epochs (int): Number of epochs for training.
        - learning_rate (float): Learning rate for optimization.
        - max_length (int): Maximum length of input sequences.
        - device (str): Device to use for training ('cpu' or 'cuda').
        - suffix (str, optional): Suffix to add to model checkpoint and confusion matrix filenames (default: '').
        """
        self.model_name = model_name  # Name of the pre-trained model
        self.num_labels = num_labels  # Number of classes
        self.batch_size = batch_size  # Batch size for training
        self.num_epochs = num_epochs  # Number of epochs for training
        self.learning_rate = learning_rate  # Learning rate for optimization
        self.max_length = max_length  # Maximum length of input sequences
        self.device = device  # Device to use for training ('cpu' or 'cuda')
        self.suffix = suffix  # Suffix for model checkpoint and confusion matrix filenames
        self.confusion_matrices_train = []  # List to store confusion matrices for training data
        self.confusion_matrices_test = []  # List to store confusion matrices for test data
        self.confusion_matrices_val = []  # List to store confusion matrices for validation data

        self._initialize_model()  # Initialize the model


    def _initialize_model(self):
        """
        Initializes the model, tokenizer, optimizer, loss function, and moves the model to the specified device.
        """
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)  # Load tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)  # Load model

        # Initialize optimizer
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        # Assign initialized objects to class attributes
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer

        # Move model to specified device
        self.model = self.model.to(self.device)

        # Define loss function
        self.loss_function = torch.nn.CrossEntropyLoss()


    def _process_data(self, df):
        """
        Processes input DataFrame into tensors for input_ids, attention_masks, and labels.

        Args:
        - df (DataFrame): Input DataFrame containing 'Sentence' and 'Labels_Encoded' columns.

        Returns:
        - TensorDataset: Tensor dataset containing processed data.
        """
        input_ids = []  # List to store input token IDs
        attention_masks = []  # List to store attention masks
        labels = []  # List to store labels
        
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            sentence = str(row['Sentence'])  # Get the sentence and ensure it's a string

            # Tokenize the sentence and convert it into tensors
            tokens = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

            input_ids.append(tokens['input_ids'][0])  # Append input token IDs
            attention_masks.append(tokens['attention_mask'][0])  # Append attention masks

            label = row['Labels_Encoded']  # Get the label
            labels.append(label)  # Append the label

        # Convert lists to tensors and move them to the specified device
        input_ids = torch.stack(input_ids).to(self.device)
        attention_masks = torch.stack(attention_masks).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        # Create a TensorDataset from the processed data
        return TensorDataset(input_ids, attention_masks, labels)


    def save_state(self, epoch, model_state, optimizer_state, path):
        """
        Saves the model and optimizer states to a checkpoint file.

        Args:
        - epoch (int): Epoch number.
        - model_state (dict): State dictionary of the model.
        - optimizer_state (dict): State dictionary of the optimizer.
        - path (str): Path to save the checkpoint file.
        """
        # Save the model and optimizer states to the specified path
        torch.save({
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer_state
        }, path)


    def load_state(self, path):
        """
        Loads model and optimizer states from a checkpoint file.

        Args:
        - path (str): Path to the checkpoint file.

        Returns:
        - int: Epoch number.
        - dict: Model state dictionary.
        - dict: Optimizer state dictionary.
        """
        # Load the checkpoint file
        checkpoint = torch.load(path)

        # Return the epoch number, model state, and optimizer state
        return checkpoint['epoch'], checkpoint['model_state'], checkpoint['optimizer_state']


    def train(self, train_df, val_df, log_dir, checkpoint_path=None):
        """
        Trains the model using the provided training and validation datasets.

        Args:
        - train_df (DataFrame): Training dataset DataFrame.
        - val_df (DataFrame): Validation dataset DataFrame.
        - log_dir (str): Directory path to save TensorBoard logs.
        - checkpoint_path (str, optional): Path to a checkpoint file to resume training (default: None).
        """
        # Initialize TensorBoard writer
        tensorboard_writer = SummaryWriter(log_dir=log_dir)

        # Process training and validation datasets
        train_dataset = self._process_data(train_df)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataset = self._process_data(val_df)
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize best validation loss
        best_val_loss = float('inf')  

        # Load checkpoint if provided
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            epoch_start, model_state, optimizer_state = self.load_state(checkpoint_path)
            self.optimizer.load_state_dict(optimizer_state)
            self.model.load_state_dict(model_state)
            print("Loaded model from " + checkpoint_path)
        else:
            epoch_start = 0
            print("Starting training from scratch")

        # Evaluate initial performance on training and validation sets
        init_loss_tr, init_accuracy_tr, init_f1_tr, init_cm_tr = self.evaluate(train_df)
        tensorboard_writer.add_scalar('Train Loss', init_loss_tr, epoch_start)
        tensorboard_writer.add_scalar('Train f1 Score', init_f1_tr, epoch_start)
        tensorboard_writer.add_scalar('Train Accuracy', init_accuracy_tr, epoch_start)
        self.save_confusion_matrix(init_cm_tr, epoch_start, "train")
        print(f"Initial Epoch {epoch_start}/{self.num_epochs} - Train Loss: {init_loss_tr} - Train Accuracy: {init_accuracy_tr:.2%}  - Train F1 Score: {init_f1_tr:.2%}")

        init_loss_val, init_accuracy_val, init_f1_val, init_cm_val = self.evaluate(val_df)
        tensorboard_writer.add_scalar('Validation Loss', init_loss_val, epoch_start)
        tensorboard_writer.add_scalar('Validation f1 Score', init_f1_val, epoch_start)
        tensorboard_writer.add_scalar('Validation Accuracy', init_accuracy_val, epoch_start)
        self.save_confusion_matrix(init_cm_val, epoch_start, "val")
        print(f"Initial Epoch {epoch_start}/{self.num_epochs} - Validation Loss: {init_loss_val} - Validation Accuracy: {init_accuracy_val:.2%}  - Validation F1 Score: {init_f1_val:.2%}")

        # Training loop
        for epoch in range(epoch_start, self.num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            correct_predictions = 0
            current_samples = 0
            all_true_labels = []
            all_predicted_labels = []
    
            for batch in tqdm(train_data_loader):
                input_ids_batch, attention_masks_batch, labels_batch = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids_batch, attention_mask=attention_masks_batch,  labels=labels_batch)
                loss = self.loss_function(outputs.logits, labels_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
    
                predicted_class = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predicted_class == labels_batch).sum().item()
                current_samples += labels_batch.size(0)

                
                all_true_labels.extend(labels_batch.cpu().numpy())
                all_predicted_labels.extend(predicted_class.cpu().numpy())
    
            accuracy = correct_predictions / current_samples
            f1 = f1_score(np.array(all_true_labels), np.array(all_predicted_labels), average='macro') 

            train_cm = confusion_matrix(np.array(all_true_labels), np.array(all_predicted_labels))
            self.confusion_matrices_train.append(train_cm)
            self.save_confusion_matrix(train_cm, epoch+1, "train")


            tensorboard_writer.add_scalar('Train Loss', total_loss / len(train_data_loader), epoch+1)
            tensorboard_writer.add_scalar('Train f1 Score', f1, epoch+1)
            tensorboard_writer.add_scalar('Train Accuracy', accuracy, epoch+1)
    
            print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {total_loss / len(train_data_loader)} - Train Accuracy: {accuracy:.2%} - Train F1 Score: {f1:.2%}")
    
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct_predictions_val = 0
            current_samples_val = 0
            all_val_true_labels = []
            all_val_predicted_labels = []
    
            with torch.no_grad():
                for val_batch in tqdm(val_data_loader):
                    val_input_ids_batch, val_attention_masks_batch, val_labels_batch = val_batch
                    val_outputs = self.model(val_input_ids_batch, attention_mask=val_attention_masks_batch, labels = val_labels_batch)
                    val_logits = val_outputs.logits
                    val_loss += self.loss_function(val_logits, val_labels_batch).item()
    
    
                    val_predicted_labels = torch.argmax(val_logits, dim=1)
                    correct_predictions_val += (val_predicted_labels == val_labels_batch).sum().item()
                    current_samples_val += val_labels_batch.size(0)
                    
                    all_val_true_labels.extend(val_labels_batch.cpu().numpy())
                    all_val_predicted_labels.extend(val_predicted_labels.cpu().numpy())
    
            val_accuracy = accuracy_score(np.array(all_val_true_labels), np.array(all_val_predicted_labels))
            val_f1 = f1_score(np.array(all_val_true_labels), np.array(all_val_predicted_labels), average='macro')

            val_cm = confusion_matrix(np.array(all_val_true_labels), np.array(all_val_predicted_labels))
            self.confusion_matrices_val.append(val_cm)
            self.save_confusion_matrix(val_cm, epoch+1, "val")

            tensorboard_writer.add_scalar('Validation Loss', val_loss / len(val_data_loader), epoch+1)
            tensorboard_writer.add_scalar('Validation f1 Score', val_f1, epoch+1)
            tensorboard_writer.add_scalar('Validation Accuracy', val_accuracy, epoch+1)
    
            print(f"Epoch {epoch+1}/{self.num_epochs} - Validation Loss: {val_loss / len(val_data_loader)} - Validation Accuracy: {val_accuracy:.2%}  - Validation F1 Score: {val_f1:.2%}")

            # Best Model Saving

            if val_loss < best_val_loss:
                # Save the model if the current validation loss is better than the previous best
                best_val_loss = val_loss
                
                checkpoint_name_best = f'multiclass_{self.suffix}_{self.model_name}_best_checkpoint.pth'
                checkpoint_path_best = os.path.join('../models_trained', checkpoint_name_best)
                
                # Explicitly close the existing file before saving the new model
                if os.path.exists(checkpoint_path_best):
                    os.remove(checkpoint_path_best)   
                    
                self.save_state(epoch+1, self.model.state_dict(), self.optimizer.state_dict(), checkpoint_path_best)

                print(f"Saved the best model {checkpoint_name_best} with validation loss: {best_val_loss / len(val_data_loader)}, Epoch: {epoch + 1}")
                print("Note: The model will be overwritten if a better model, based on the validation loss, is found.")


            # # Fifth Model Saving

            # if (epoch+1) % 5 == 0:         

            #     checkpoint_name_fifth = f'multiclass_{self.suffix}_{self.model_name}_epoch{epoch+1}_checkpoint.pth'
            #     checkpoint_path_fifth = os.path.join('../models_trained', checkpoint_name_fifth)

            #     # Explicitly close the existing file before saving the new model
            #     if os.path.exists(checkpoint_path_fifth):
            #         os.remove(checkpoint_path_fifth)

            #     self.save_state(epoch+1, self.model.state_dict(), self.optimizer.state_dict(), checkpoint_path_fifth)
            #     print(f"Saved model {checkpoint_name_fifth} with validation loss: {val_loss / len(val_data_loader)}, Epoch: {epoch + 1}")

        # Final Model Saving
            
        checkpoint_name_final = f'multiclass_{self.suffix}_{self.model_name}_epoch{epoch+1}_checkpoint.pth'
        checkpoint_path_final = os.path.join('../models_trained', checkpoint_name_final)

        # Explicitly close the existing file before saving the new model
        if os.path.exists(checkpoint_path_final):
            os.remove(checkpoint_path_final)

        self.save_state(epoch+1, self.model.state_dict(), self.optimizer.state_dict(), checkpoint_path_final)
        print(f"Saved the final model {checkpoint_name_final} with validation loss: {val_loss / len(val_data_loader)}, Epoch: {self.num_epochs}")
         
        tensorboard_writer.close()
              

    def save_confusion_matrix(self, cm, epoch, mode):
        """
        Saves the confusion matrix to a file.

        Args:
        - cm (array): Confusion matrix array.
        - epoch (int): Epoch number.
        - mode (str): Mode of confusion matrix ('train', 'val', 'test').
        """
        # Construct the filename
        cm_file_name = f"{mode}_cm_multiclass_{self.suffix}_{self.model_name}_epoch{epoch}"
        # Define the full file path
        cm_path = os.path.join('../reports/figures', cm_file_name)
        # Save the confusion matrix to a file
        with open(cm_path, 'wb') as file:
            pickle.dump(cm, file)


    def save_confusion_matrices(self):
        """
        Saves training, validation, and test confusion matrices to files.
        """
        # Define file names for confusion matrices
        cm_train_file_name =  f"training_cm_multiclass_{self.suffix}_{self.model_name}_{self.num_epochs}epochs"
        cm_val_file_name = f"validation_cm_multiclass_{self.suffix}_{self.model_name}_{self.num_epochs}epochs"
        cm_test_file_name = f"testing_cm_multiclass_{self.suffix}_{self.model_name}_{self.num_epochs}epochs"
        
        # Define file paths for confusion matrices
        cm_train_path = os.path.join('../reports/figures', cm_train_file_name)
        cm_val_path = os.path.join('../reports/figures', cm_val_file_name)
        cm_test_path = os.path.join('../reports/figures', cm_test_file_name)

        # Remove existing files if present
        if os.path.exists(cm_train_path):
            os.remove(cm_train_path)
        if os.path.exists(cm_val_path):
            os.remove(cm_val_path)
        if os.path.exists(cm_test_path):
            os.remove(cm_test_path)
        
        # Save training confusion matrix
        with open(cm_train_path, 'wb') as file:
            pickle.dump(self.confusion_matrices_train, file)
        # Save validation confusion matrix
        with open(cm_val_path, 'wb') as file:
            pickle.dump(self.confusion_matrices_val, file)
        # Save test confusion matrix
        with open(cm_test_path, 'wb') as file:
            pickle.dump(self.confusion_matrices_test, file)


    def load_model(self, checkpoint_path):
        """
        Loads the model from a checkpoint file.

        Args:
        - checkpoint_path (str): Path to the checkpoint file.
        """
        # Load model state from the checkpoint file
        _, model_state, _ = self.load_state(checkpoint_path)
        # Load model state into the model
        self.model.load_state_dict(model_state)


    def evaluate(self, test_df, mode='train'):
        """
        Evaluates the model on the test dataset.

        Args:
        - test_df (DataFrame): Test dataset DataFrame.
        - mode (str, optional): Evaluation mode ('train' or 'test') (default: 'train').

        Returns:
        - float: Average loss.
        - float: Accuracy.
        - float: F1 score.
        - array: Confusion matrix.
        """
        # Process the test dataset
        test_dataset = self._process_data(test_df)
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Set the model to evaluation mode
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        all_test_true_labels = []
        all_test_predicted_labels = []

        # Disable gradient computation during evaluation
        with torch.no_grad():
            for batch in test_data_loader:
                test_input_ids_batch, test_attention_masks_batch, test_labels_batch = batch

                # Forward pass
                outputs = self.model(test_input_ids_batch, attention_mask=test_attention_masks_batch, labels=test_labels_batch)

                # Accumulate loss
                total_loss += outputs.loss.item()

                # Calculate predicted class
                predicted_class = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predicted_class == test_labels_batch).sum().item()
                total_samples += test_labels_batch.size(0)

                # Collect true and predicted labels
                all_test_true_labels.extend(test_labels_batch.cpu().numpy())
                all_test_predicted_labels.extend(predicted_class.cpu().numpy())

        # Calculate confusion matrix
        test_cm = confusion_matrix(np.array(all_test_true_labels), np.array(all_test_predicted_labels))
        self.confusion_matrices_test.append(test_cm)

        # Calculate accuracy, loss, and F1 score
        accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(test_data_loader)
        test_f1 = f1_score(np.array(all_test_true_labels), np.array(all_test_predicted_labels), average='macro')

        # Print evaluation metrics if in test mode
        if mode == 'test':
            print(f"\nTest Set - Average Loss: {average_loss:.2}")
            print(f"\nTest Set Accuracy: {accuracy:.2%}")
            print(f"\nTest Set F1 Score: {test_f1:.2%}")

        return average_loss, accuracy, test_f1, test_cm

    
    def single_inference(self, checkpoint_path, train_df, sentence):
        """
        Performs single inference on a given sentence.

        Args:
        - checkpoint_path (str): Path to the checkpoint file.
        - train_df (DataFrame): DataFrame containing label mappings.
        - sentence (str): Input sentence for inference.

        Returns:
        - dict: Dictionary containing predicted labels and probabilities.
        """
        # Load model state and initialize tokenizer
        _, model_state, _ = self.load_state(checkpoint_path)
        model = self.model
        model.config.id2label = train_df.set_index('Labels_Encoded')['Labels'].to_dict()
        model.load_state_dict(model_state)
        tokenizer = self.tokenizer
        
        # Initialize text classification pipeline
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=3)
        
        # Perform inference
        return pipe(sentence)
        

    def print_cm(self, mode, df):
        """
        Prints the confusion matrix.

        Args:
        - mode (str): Mode of confusion matrix ('train', 'val', 'test').
        - df (DataFrame): DataFrame containing label mappings.
        """
        # Create a dictionary mapping label indices to their corresponding class names
        label_to_class_val = {}
        for i in range(self.num_labels):
            label_to_class_val[i] = df[df['Labels_Encoded'] == i].iloc[0]['Labels']
            
        # Define x and y axis labels for the confusion matrix
        x_axis_labels = [label_to_class_val[i] for i in range(self.num_labels)]
        y_axis_labels = [label_to_class_val[i] for i in range(self.num_labels)]
    
        if mode == "train":
            epoch = 0
            num_matrices = len(self.confusion_matrices_train)
            plt.figure(figsize=(15, 10 * num_matrices))
            
            # Plot each training confusion matrix
            for i in self.confusion_matrices_train:
                epoch += 1
                plt.subplot(num_matrices, 1, epoch)
                sns.heatmap(i, annot=True, cmap='Blues', fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
                plt.title(f'Training Confusion Matrix - Epoch {epoch}')
                plt.xlabel('Predicted labels')
                plt.ylabel('True labels')

            plt.tight_layout()
            plt.show()
            
        elif mode == "val":
            epoch = 0
            num_matrices = len(self.confusion_matrices_val)
            plt.figure(figsize=(15, 10 * num_matrices))
            
            # Plot each validation confusion matrix
            for i in self.confusion_matrices_val:
                epoch += 1
                plt.subplot(num_matrices, 1, epoch)
                sns.heatmap(i, annot=True, cmap='Blues', fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
                plt.title(f'Validation Confusion Matrix - Epoch {epoch}')
                plt.xlabel('Predicted labels')
                plt.ylabel('True labels')

            plt.tight_layout()
            plt.show()
            
        elif mode == "test":
            # Assuming there's only one matrix for the test mode
            plt.figure(figsize=(15, 10))
            confusion_matrix_test = self.confusion_matrices_test[0]
            sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
            plt.title('Test Confusion Matrix')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.show()
            
        else:
            print("Unsupported argument")


