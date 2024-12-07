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
from transformers import pipeline
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_bert_ner")
model = AutoModelForTokenClassification.from_pretrained("./fine_tuned_bert_ner")

# Load the fine-tuned model and tokenizer
ner_pipeline = pipeline("ner", model=model, device=device,tokenizer=tokenizer, aggregation_strategy="simple")





# Test the pipeline
text = "Mount Everest is one of the tallest peaks in the world."
result = ner_pipeline(text)
print(result)
