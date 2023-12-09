import pandas as pd
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import concurrent.futures
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing, SAGEConv
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch_geometric.utils import from_networkx

# Current working directory
pwd

# List directories
os.listdir('michigan_patients/csv')

# Load data from directories
directory = '/mnt/ufs18/home-230/pandavis/CSE881 Data Mining/Data_Mining_Project/michigan_patients/csv'
df = {}
df_type = []
synthea_files = [file for file in os.listdir(directory) if str(file).endswith('.csv')]
for file in synthea_files:
    file_path = os.path.join(directory, file)
    df[file.split('.')[0]] = pd.read_csv(file_path)
    df_type.append(file.split('.')[0])

# DataFrames
patients = df['patients']
observations = df['observations']
encounters = df['encounters']
procedures = df['procedures']
supplies = df['supplies']
imaging_studies = df['imaging_studies']
conditions = df['conditions']
careplans = df['careplans']
devices = df['devices']
medications = df['medications']
allergies = df['allergies']
immunizations = df['immunizations']
providers = df['providers']

# DataFrame types
df_type

# Display DataFrame information
for type_ in df_type:
    columns = ', '.join(df[type_].columns)
    length = len(df[type_])
    print(f"{type_} columns:\n{columns}\nLength of df :{length}\n")

# Display head of 'supplies' DataFrame
supplies.head()

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")
model = AutoModel.from_pretrained("microsoft/Multilingual-MiniLM-L12-H384")

# List of DataFrames
df_list = [procedures, supplies, encounters, conditions, careplans, devices, medications, allergies, immunizations]

# Generate embeddings
def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]

# Unique descriptions
unique_descriptions = set()
for df in df_list:
    for _, row in df.iterrows():
        desc = row.get('DESCRIPTION', '')
        if 'REASONDESCRIPTION' in df.columns and pd.notna(row['REASONDESCRIPTION']):
            combined_desc = str(desc) + " " + str(row['REASONDESCRIPTION'])
        else:
            combined_desc = desc
        unique_descriptions.add(combined_desc)

# Length of unique descriptions
len(unique_descriptions)

# Embeddings dictionary
embeddings_dict = {desc: generate_embeddings(desc) for desc in unique_descriptions}

# Filter encounters
encounter_counts = encounters['PATIENT'].value_counts()
patients_less_than_100_encounters = encounter_counts[encounter_counts < 25].index
filtered_encounters = encounters[encounters['PATIENT'].isin(patients_less_than_100_encounters)]

# Create graph
G = nx.Graph()
for patient_id in encounters['PATIENT'].unique():
    G.add_node(patient_id, type='patient', embedding=np.zeros(384))
for _, row in encounters.iterrows():
    encounter_id = row['Id']
    desc = row['DESCRIPTION']
    embedding = embeddings_dict.get(desc, np.zeros(model.config.hidden_size))
    G.add_node(encounter_id, type='encounter', embedding=embedding)
for desc in list(embeddings_dict.keys()):
    G.add_node(desc, type='description', embedding=embeddings_dict.get(desc, np.zeros(model.config.hidden_size)))
for _, row in encounters.iterrows():
    G.add_edge(row['PATIENT'], row['Id'])
for df in df_list:
    if df is not encounters:
       for _, row in df.iterrows():
            encounter_id = row['ENCOUNTER']
            desc = row.get('DESCRIPTION', '')
            reason_desc = row.get('REASONDESCRIPTION', '')
            combined_desc = desc if pd.isna(reason_desc) else str(desc) + " " + str(reason_desc)
            if combined_desc in unique_descriptions:
                G.add_edge(encounter_id, combined_desc)