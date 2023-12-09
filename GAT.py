import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GATConv

# Load the graph
G = pickle.load(open('graph_G_with_embeddings.pkl', 'rb'))

# Add missing attributes to nodes
all_attrs = set()
for _, node_data in G.nodes(data=True):
    all_attrs.update(node_data.keys())

for _, node_data in G.nodes(data=True):
    for attr in all_attrs:
        if attr not in node_data:
            node_data[attr] = np.zeros(384)

# Convert to PyG format
pyg_formatted_graph = from_networkx(G)

# Node to index mapping
node_to_index = {node: idx for idx, node in enumerate(G.nodes())}

# Create label mapping functions
def create_label_mapping(dataframe, description_column, reason_column):
    node_label_mapping = {}
    for _, row in dataframe.iterrows():
        desc = row.get(description_column, '')
        reason_desc = row.get(reason_column, None)
        label = str(desc) if pd.isna(reason_desc) else f"{desc} {reason_desc}"
        node_label_mapping[label] = label
    return node_label_mapping

# Label mappings for various categories
medication_label_mapping = create_label_mapping(medications, 'DESCRIPTION', 'REASONDESCRIPTION')
immunization_label_mapping = create_label_mapping(immunizations, 'DESCRIPTION', 'REASONDESCRIPTION')
careplan_label_mapping = create_label_mapping(careplans, 'DESCRIPTION', 'REASONDESCRIPTION')

# Label encoders
medication_le = LabelEncoder()
immunization_le = LabelEncoder()
careplan_le = LabelEncoder()

# Fit label encoders
medication_labels = list(medication_label_mapping.values()) + [-1]
immunization_labels = list(immunization_label_mapping.values()) + [-1]
careplan_labels = list(careplan_label_mapping.values()) + [-1]
medication_le.fit(medication_labels)
immunization_le.fit(immunization_labels)
careplan_le.fit(careplan_labels)

# Encode labels
encoded_labels = np.full((len(G.nodes()), 3), -1, dtype=int)
for node, index in node_to_index.items():
    if node in medication_label_mapping:
        medication_label = medication_label_mapping[node]
        encoded_labels[index, 0] = medication_le.transform([medication_label])[0]
    if node in immunization_label_mapping:
        immunization_label = immunization_label_mapping[node]
        encoded_labels[index, 1] = immunization_le.transform([immunization_label])[0]
    if node in careplan_label_mapping:
        careplan_label = careplan_label_mapping[node]
        encoded_labels[index, 2] = careplan_le.transform([careplan_label])[0]

label_tensor = torch.tensor(encoded_labels, dtype=torch.long)

# Node embeddings and edge indices
node_embeddings = [G.nodes[node]['embedding'] for node in G.nodes()]
node_embeddings_tensor = torch.stack([torch.tensor(e, dtype=torch.float) for e in node_embeddings])
edge_index_list = [(node_to_index[u], node_to_index[v]) for u, v in G.edges()]
edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

# Update PyG graph with new data
pyg_formatted_graph = Data(x=node_embeddings_tensor, edge_index=edge_index_tensor, y=label_tensor)

# MultiTask GAT model definition
class MultiTaskGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes_per_task):
        super(MultiTaskGAT, self).__init__()
        self.hidden_units = 16
        self.num_heads = 8
        # Common layers
        self.conv1 = GATConv(num_node_features, self.hidden_units, heads=self.num_heads, dropout=0.6)
        self.conv2 = GATConv(self.hidden_units * self.num_heads, self.hidden_units, heads=self.num_heads, dropout=0.6)
        # Separate output layers for each task
        self.out_medication = GATConv(self.hidden_units * self.num_heads, num_classes_per_task[0], heads=1, dropout=0.6)
        self.out_immunization = GATConv(self.hidden_units * self.num_heads, num_classes_per_task[1], heads=1, dropout=0.6)
        self.out_careplan = GATConv(self.hidden_units * self.num_heads, num_classes_per_task[2], heads=1, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Common layers
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        # Task-specific layers
        medication_logits = self.out_medication(x, edge_index)
        immunization_logits = self.out_immunization(x, edge_index)
        careplan_logits = self.out_careplan(x, edge_index)
        return medication_logits, immunization_logits, careplan_logits



# Define encounter indices and split data
encounter_indices = encounters['Id'].apply(lambda x: node_to_index[x])
num_encounters = len(encounter_indices)
num_train = int(0.6 * num_encounters)
num_val = int(0.2 * num_encounters)

# Convert encounter_indices to a tensor
encounter_indices_tensor = torch.tensor(encounter_indices, dtype=torch.long)
shuffled_encounter_indices = encounter_indices_tensor[torch.randperm(num_encounters)]

# Split indices for training, validation, and testing
train_indices = shuffled_encounter_indices[:num_train]
val_indices = shuffled_encounter_indices[num_train:num_train + num_val]
test_indices = shuffled_encounter_indices[num_train + num_val:]

# Masks for train, val, and test
total_nodes = pyg_formatted_graph.num_nodes
train_mask = torch.zeros(total_nodes, dtype=torch.bool)
val_mask = torch.zeros(total_nodes, dtype=torch.bool)
test_mask = torch.zeros(total_nodes, dtype=torch.bool)
train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

# Update PyG graph with masks
pyg_formatted_graph.train_mask = train_mask
pyg_formatted_graph.val_mask = val_mask
pyg_formatted_graph.test_mask = test_mask

# Model and optimizer
model = MultiTaskGAT(num_node_features=pyg_formatted_graph.num_node_features, num_classes_per_task=[400, 25, 93])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

# Training labels
medication_labels = pyg_formatted_graph.y[:, 0]
immunization_labels = pyg_formatted_graph.y[:, 1]
careplan_labels = pyg_formatted_graph.y[:, 2]

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    medication_logits, immunization_logits, careplan_logits = model(pyg_formatted_graph)
    valid_medication_mask = (medication_labels[train_mask] != -1)
    valid_immunization_mask = (immunization_labels[train_mask] != -1)
    valid_careplan_mask = (careplan_labels[train_mask] != -1)
    loss_medication = F.cross_entropy(medication_logits[train_mask][valid_medication_mask], medication_labels[train_mask][valid_medication_mask])
    loss_immunization = F.cross_entropy(immunization_logits[train_mask][valid_immunization_mask], immunization_labels[train_mask][valid_immunization_mask])
    loss_careplan = F.cross_entropy(careplan_logits[train_mask][valid_careplan_mask], careplan_labels[train_mask][valid_careplan_mask])
    total_loss = loss_medication + loss_immunization + loss_careplan
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

# Evaluation function
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        medication_logits, immunization_logits, careplan_logits = model(pyg_formatted_graph)
        med_preds = medication_logits[mask].max(1)[1]
        imm_preds = immunization_logits[mask].max(1)[1]
        cp_preds = careplan_logits[mask].max(1)[1]
        true_med_labels = medication_labels[mask]
        true_imm_labels = immunization_labels[mask]
        true_cp_labels = careplan_labels[mask]
    return (true_med_labels.cpu(), med_preds.cpu(), true_imm_labels.cpu(), imm_preds.cpu(), true_cp_labels.cpu(), cp_preds.cpu())

# Predict function for encounters
def predict_for_encounter(encounter_index):
    model.eval()
    with torch.no_grad():
        medication_logits, immunization_logits, careplan_logits = model(pyg_formatted_graph)
        predicted_med = medication_logits[encounter_index].argmax(dim=0).item()
        predicted_imm = immunization_logits[encounter_index].argmax(dim=0).item()
        predicted_cp = careplan_logits[encounter_index].argmax(dim=0).item()
    return predicted_med, predicted_imm, predicted_cp

# Calculate accuracy
def calculate_accuracy(true_labels, preds):
    correct = preds.eq(true_labels).sum().item()
    acc = correct / len(true_labels)
    return acc

# Training loop
for epoch in range(50):
    loss = train()
    true_med_train, med_preds_train, true_imm_train, imm_preds_train, true_cp_train, cp_preds_train = evaluate(train_mask)
    train_med_acc = calculate_accuracy(true_med_train, med_preds_train)
    train_imm_acc = calculate_accuracy(true_imm_train, imm_preds_train)
    train_cp_acc = calculate_accuracy(true_cp_train, cp_preds_train)
    true_med_val, med_preds_val, true_imm_val, imm_preds_val, true_cp_val, cp_preds_val = evaluate(val_mask)
    val_med_acc = calculate_accuracy(true_med_val, med_preds_val)
    val_imm_acc = calculate_accuracy(true_imm_val, imm_preds_val)
    val_cp_acc = calculate_accuracy(true_cp_val, cp_preds_val)
    print(f'Epoch: {epoch+1}, Loss: {loss:.4f}')
    print(f'Train Acc (Medication): {train_med_acc:.4f}, Immunization: {train_imm_acc:.4f}, CarePlan: {train_cp_acc:.4f}')
    print(f'Val Acc (Medication): {val_med_acc:.4f}, Immunization: {val_imm_acc:.4f}, CarePlan: {val_cp_acc:.4f}')

# Testing and evaluation
true_med_test, med_preds_test, true_imm_test, imm_preds_test, true_cp_test, cp_preds_test = evaluate(test_mask)
valid_med_test_mask = true_med_test != -1
valid_imm_test_mask = true_imm_test != -1
valid_cp_test_mask = true_cp_test != -1
accuracy_med = accuracy_score(true_med_test[valid_med_test_mask], med_preds_test[valid_med_test_mask]) if valid_med_test_mask.sum() > 0 else 0
accuracy_imm = accuracy_score(true_imm_test[valid_imm_test_mask], imm_preds_test[valid_imm_test_mask]) if valid_imm_test_mask.sum() > 0 else 0
accuracy_cp = accuracy_score(true_cp_test[valid_cp_test_mask], cp_preds_test[valid_cp_test_mask]) if valid_cp_test_mask.sum() > 0 else 0
print(f'Test Accuracy for Medication: {accuracy_med:.3f}')
print(f'Test Accuracy for Immunization: {accuracy_imm:.3f}')
print(f'Test Accuracy for CarePlan: {accuracy_cp:.3f}')




# Define an encounter ID and retrieve its index
encounter_id = '4870f6c0-1fe8-3d17-51b1-297df0d00370'
encounter_index = node_to_index[encounter_id]

# Predict for a specific encounter
predicted_med, predicted_imm, predicted_cp = predict_for_encounter(encounter_index)
med_label = medication_le.inverse_transform([predicted_med])[0]
imm_label = immunization_le.inverse_transform([predicted_imm])[0]
cp_label = careplan_le.inverse_transform([predicted_cp])[0]

# Display predictions
print(f"Predicted Medication Class: {med_label}")
print(f"Predicted Immunization Class: {imm_label}")
print(f"Predicted CarePlan Class: {cp_label}")

# Predict for another encounter (if needed)
predicted_med, predicted_imm, predicted_cp = predict_for_encounter(encounter_index)
med_label = medication_le.inverse_transform([predicted_med])[0]
imm_label = immunization_le.inverse_transform([predicted_imm])[0]
cp_label = careplan_le.inverse_transform([predicted_cp])[0]

# Display predictions
print(f"Predicted Medication Class: {med_label}")
print(f"Predicted Immunization Class: {imm_label}")
print(f"Predicted CarePlan Class: {cp_label}")

# Define function to predict for an encounter
def predict_for_encounter(encounter_index):
    model.eval()
    with torch.no_grad():
        logits = model(pyg_formatted_graph)
        encounter_logits = logits[encounter_index]
        probabilities = F.softmax(encounter_logits, dim=0)
        predicted_classes = probabilities.argmax(dim=0)
    return predicted_classes, probabilities

# Example usage of the prediction function
encounter_index = 50000
predicted_classes, probabilities = predict_for_encounter(encounter_index)
print(f"Predicted Classes: {predicted_classes}, Probabilities: {probabilities}")

