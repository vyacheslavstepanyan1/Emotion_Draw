# **Dataset Used for Fine-Tuning BERT and Friends**

## [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?resource=download) 

### **About Dataset**

The Emotions Dataset for NLP is a collection of documents annotated with their corresponding emotions, providing valuable resources for natural language processing (NLP) classification tasks. 📚🔍 It comprises lists of documents paired with emotion labels and is split into train, test, and validation sets to facilitate the development of machine learning models. 🛠️💻

### **Example**

An example entry from the dataset follows the format:

```
"I feel like I am still looking at a blank canvas blank pieces of paper"; sadness
```

### **Sizes of The Sets: 80-20-20 (%)**

* **Training Set** : 16,000
* **Validation Set** : 2000
* **Test Set** : 2000

### **Acknowledgements**

This dataset is made available thanks to [Elvis](https://lnkd.in/eXJ8QVB) and the Hugging Face team. The methodology used to prepare the dataset is detailed in the following publication: [CARER: Contextualized Affect Representations for Emotion Recognition](https://www.aclweb.org/anthology/D18-1404/).

### **Inspiration**

The Kaggle Emotion Dataset serves as a valuable resource for the community, enabling the development of emotion classification models using NLP-based approaches. 🌟📊 Researchers and practitioners can leverage this dataset to explore a variety of questions related to sentiment analysis and mood identification, such as:

- What is the sentiment of a customer's comment?

- What is the mood associated with today's special food?

### **Labels**

The dataset includes six emotion labels: `Anger`, `Joy`, `Love`, `Fear`, `Sadness`, and `Surprise`. Each document in the dataset is annotated with one of these emotions.

### **Limitations**

However, sadly, all sentences in the dataset follow a specific format, starting with "I am ..." or "I...". While this format simplifies the annotation process, it may limit the effectiveness of fine-tuned models in extracting sentiment from inputs of different formats. 🤔📝

**Therefore, we also plan to explore alternative datasets in the future.**

***Note:** You can find these datasets in `/Emotion_Draw/Emotion_Draw/bert_part/data/raw/`.*

## **Preprocessing**

***Note:** You can find an associated notebook in `/Emotion_Draw/Emotion_Draw/bert_part/notebooks/data_creation.ipynb`.*

**Data Processing for Training Dataset**

This script explains the data processing steps for the training dataset. 

--------------------

### **Reading and Displaying the Dataset**

Initially, the raw text file is read, and its contents are split into sentences and labels. 

```python
#Read the text file
with open('../data/raw/train.txt', 'r') as file:
    lines = file.readlines()

#Split each line by semicolon
data = [line.strip().split(';') for line in lines]

#Create DataFrame
train_df = pd.DataFrame(data, columns=['Sentence', 'Labels'])

#Display the DataFrame
train_df
```

--------------------

The DataFrame is then created and displayed. Following this, exploratory data analysis (EDA) is conducted, including descriptive statistics, checking for missing values, examining data types, and identifying unique labels.

### **Dataset Exploration**

```python
# Descriptive statistics
train_df.describe()

# Check for missing values
train_df.isnull().sum()

# Data types of columns
train_df.dtypes

# Unique labels
train_df['Labels'].unique()
```

--------------------

Next, label encoding is performed to convert categorical labels into numerical values.

### **Label Encoding**

```python
# Encode labels
train_df['Labels_Encoded'] = label_encoder.fit_transform(train_df['Labels'])
train_df
```

--------------------

Finally, the processed dataset is saved into a CSV file for further use in model training.

### **Saving Dataset into CSV**

```python
# Specify the file path where you want to save the CSV file
file_path = '../data/processed/train_data.csv'

# Save the DataFrame to a CSV file
train_df.to_csv(file_path, index=False)
```

--------------------

Similar procedures are applied for validation and test sets. Adjustments to file paths and other configurations may be necessary based on your specific setup.