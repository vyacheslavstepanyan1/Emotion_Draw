import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm


class MulticlassClassificationTrainer:
    def __init__(self, model_name, num_labels, batch_size, num_epochs, learning_rate, max_length, device, suffix=""):
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.device = device
        self.suffix = suffix
        self.confusion_matrices_train = []
        self.confusion_matrices_test = []
        self.confusion_matrices_val = []

        self._initialize_model()

    def _initialize_model(self):
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

        optimizer = AdamW(model.parameters(), lr=self.learning_rate)

        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = optimizer

        self.model = self.model.to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def _process_data(self, df):
        input_ids = []
        attention_masks = []
        labels = []
        
        for index, row in df.iterrows():
            sentence = str(row['Sentence'])  # Ensure it's a string

            tokens = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

            input_ids.append(tokens['input_ids'][0])
            attention_masks.append(tokens['attention_mask'][0])

            label = row['Labels_Encoded']
            labels.append(label)

        input_ids = torch.stack(input_ids).to(self.device)
        attention_masks = torch.stack(attention_masks).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        return TensorDataset(input_ids, attention_masks, labels)

    def save_state(self, epoch, model_state, optimizer_state, path):
        torch.save({
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer_state
        }, path)

    def load_state(self, path):
        checkpoint = torch.load(path)
        return checkpoint['epoch'], checkpoint['model_state'], checkpoint['optimizer_state']

    def train(self, train_df, val_df, log_dir, checkpoint_path=None):
        tensorboard_writer = SummaryWriter(log_dir=log_dir)
        train_dataset = self._process_data(train_df)
        train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
        val_dataset = self._process_data(val_df)
        val_data_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        best_val_loss = float('inf')  

        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            epoch_start, model_state, optimizer_state = self.load_state(checkpoint_path)
            self.optimizer.load_state_dict(optimizer_state)
            self.model.load_state_dict(model_state)
            print("Loaded model from " + checkpoint_path)
        else:
            epoch_start = 0
            print("Starting training from scratch")

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
        cm_file_name = f"{mode}_cm_multiclass_{self.suffix}_{self.model_name}_epoch{epoch}"
        cm_path = os.path.join('../reports/figures', cm_file_name)
        with open(cm_path, 'wb') as file:
            pickle.dump(cm, file)


    def save_confusion_matrices(self):

        cm_train_file_name =  f"training_cm_multiclass_{self.suffix}_{self.model_name}_{self.num_epochs}epochs"
        cm_val_file_name = f"validation_cm_multiclass_{self.suffix}_{self.model_name}_{self.num_epochs}epochs"
        cm_test_file_name = f"testing_cm_multiclass_{self.suffix}_{self.model_name}_{self.num_epochs}epochs"
        
        cm_train_path = os.path.join('../reports/figures', cm_train_file_name)
        cm_val_path = os.path.join('../reports/figures', cm_val_file_name)
        cm_test_path = os.path.join('../reports/figures', cm_test_file_name)

        if os.path.exists(cm_train_path):
            os.remove(cm_train_path)  # Remove the directory

        if os.path.exists(cm_val_path):
            os.remove(cm_val_path)  # Remove the directory
            
        if os.path.exists(cm_test_path):
            os.remove(cm_test_path)  # Remove the directory
        
        with open(cm_train_path, 'wb') as file:
            pickle.dump(self.confusion_matrices_train, file)
        with open(cm_val_path, 'wb') as file:
            pickle.dump(self.confusion_matrices_val, file)
        with open(cm_test_path, 'wb') as file:
            pickle.dump(self.confusion_matrices_test, file)

    def load_model(self, checkpoint_path):
        _, model_state , _ = self.load_state(checkpoint_path)
        self.model.load_state_dict(model_state)

    def evaluate(self, test_df, mode = 'train'):
        test_dataset = self._process_data(test_df)
        test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        all_test_true_labels = []
        all_test_predicted_labels = []

        with torch.no_grad():
            for batch in test_data_loader:
                test_input_ids_batch, test_attention_masks_batch, test_labels_batch = batch

                outputs = self.model(test_input_ids_batch, attention_mask=test_attention_masks_batch, labels = test_labels_batch)

                total_loss += outputs.loss.item()

                predicted_class = torch.argmax(outputs.logits, dim=1)
                correct_predictions += (predicted_class == test_labels_batch).sum().item()
                total_samples += test_labels_batch.size(0)

                all_test_true_labels.extend(test_labels_batch.cpu().numpy())
                all_test_predicted_labels.extend(predicted_class.cpu().numpy())


        test_cm = confusion_matrix(np.array(all_test_true_labels), np.array(all_test_predicted_labels))
        self.confusion_matrices_test.append(test_cm)

        # Calculate accuracy and loss
        accuracy = correct_predictions / total_samples
        average_loss = total_loss / len(test_data_loader)
        test_f1 = f1_score(np.array(all_test_true_labels), np.array(all_test_predicted_labels), average='macro')

        if mode == 'test':
            print(f"\nTest Set - Average Loss: {average_loss:.2}")
            print(f"\nTest Set Accuracy: {accuracy:.2%}")
            print(f"\nTest Set F1 Score: {test_f1:.2%}")

        return average_loss, accuracy, test_f1, test_cm
    

    def single_inference(self, checkpoint_path, train_df, sentence):
        _, model_state , _ = self.load_state(checkpoint_path)
        model = self.model
        model.config.id2label = train_df.set_index('Labels_Encoded')['Labels'].to_dict()
        model.load_state_dict(model_state)
        tokenizer = self.tokenizer
        pipe = pipeline("text-classification", model = model, tokenizer=tokenizer, top_k=3)
        return pipe(sentence)
        

    def print_cm(self, mode, df):

        label_to_class_val = {}
        for i in range(self.num_labels):
            label_to_class_val[i] = df[df['Labels_Encoded'] == i].iloc[0]['Labels']
            
        x_axis_labels = [label_to_class_val[i] for i in range(self.num_labels)]
        y_axis_labels = [label_to_class_val[i] for i in range(self.num_labels)]
    
        if mode == "train":
            epoch = 0
            num_matrices = len(self.confusion_matrices_train)
            plt.figure(figsize=(15, 10 * num_matrices))
            
            for i in self.confusion_matrices_train:
                epoch+=1
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
            
            for i in self.confusion_matrices_val:
                epoch+=1
                plt.subplot(num_matrices, 1, epoch)
                sns.heatmap(i, annot=True, cmap='Blues', fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
                plt.title(f'Validation Confusion Matrix - Epoch {epoch}')
                plt.xlabel('Predicted labels')
                plt.ylabel('True labels')

            plt.tight_layout()
            plt.show()
            
        elif mode == "test":
            plt.figure(figsize=(15, 10))
            confusion_matrix_test = self.confusion_matrices_test[0]  # Assuming there's only one matrix
            sns.heatmap(confusion_matrix_test, annot=True, cmap='Blues', fmt='d', xticklabels=x_axis_labels, yticklabels=y_axis_labels)
            plt.title('Test Confusion Matrix')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.show()
            
        else:
            print("Unsupported argument")

