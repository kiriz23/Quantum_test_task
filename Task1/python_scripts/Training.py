import torch, os
import pandas as pd
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import Dataset
import accelerate
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader, Dataset
import accelerate
import transformers
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

def load_bio_dataset_to_dataframe(file_path):
    data = []
    sentence_id = 0

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:  # End of a sentence
                sentence_id += 1
            else:
                token, label = line.split()
                data.append({"sentence_id": sentence_id, "token": token, "label": label})

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Load dataset into a DataFrame
file_path = "D:\\InterviewProject\\Task1\\dataset\\improved_data.txt"  # Replace with your file path
df = load_bio_dataset_to_dataframe(file_path)

labels = df['label'].unique().tolist()
labels = [s.strip() for s in labels ]

NUM_LABELS= len(labels)

id2label={id:label for id,label in enumerate(labels)}

label2id={label:id for id,label in enumerate(labels)}

df["ids"]=df.label.map(lambda x: label2id[x.strip()])




def prepare_data(df):
    sentences = df.groupby("sentence_id")["token"].apply(list).tolist()
    labels = df.groupby("sentence_id")["label"].apply(list).tolist()
    return sentences, labels

# Prepare tokens and labels
sentences, labels = prepare_data(df)

# Split the dataset into train and test sets
train_sentences, test_sentences, train_labels, test_labels = train_test_split(
    sentences, labels, test_size=0.2, random_state=42
)


# Load BERT tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize and align labels
def tokenize_and_align_labels(sentences, labels):
    tokenized_inputs = tokenizer(
        sentences,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to word indices
        label_ids = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_id != previous_word_id:
                label_ids.append(label2id[label[word_id]])  # Assign label to first subword
            else:
                label_ids.append(-100)  # Ignore other subword parts
            previous_word_id = word_id
        aligned_labels.append(label_ids)
    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

# Tokenize train and test datasets
train_inputs = tokenize_and_align_labels(train_sentences, train_labels)
test_inputs = tokenize_and_align_labels(test_sentences, test_labels)

# Load pre-trained BERT model
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label2id)  # Number of unique labels
)

class NERDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}

# Create DataLoader-compatible datasets
train_dataset = NERDataset(train_inputs)
test_dataset = NERDataset(test_inputs)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

model.save_pretrained("./fine_tuned_bert_ner")
tokenizer.save_pretrained("./fine_tuned_bert_ner")



#Evaluate the model
mlb = MultiLabelBinarizer()


# Predictions
predictions, labels, _ = trainer.predict(test_dataset)
predictions = np.argmax(predictions, axis=2)

# Convert IDs to tags
true_tags = [[id2label[label_id] for label_id in sentence if label_id != -100] for sentence in labels]
pred_tags = [[id2label[pred_id] for pred_id, label_id in zip(sentence, labels[i]) if label_id != -100] for i, sentence in enumerate(predictions)]

# Flatten the lists of true and predicted tags
flat_true_tags = [tag for sentence in true_tags for tag in sentence]
flat_pred_tags = [tag for sentence in pred_tags for tag in sentence]

# Print the classification report
print(classification_report(flat_true_tags, flat_pred_tags,zero_division=0))
print("Training is complete")