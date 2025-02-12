import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Configurações gerais
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs_distillation = 1000  # Número de épocas para destilação
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Importação dos dados
dataset_url = "hf://datasets/Goastro/mlx-grpo-dataset/data/train-00000-of-00001.parquet"
dtrain = pd.read_parquet(dataset_url)

# Pré-processamento dos dados
def preprocess_text(text):
    """Remove espaços extras e converte para lower case."""
    return " ".join(text.lower().strip().split())

dtrain['prompt'] = dtrain['prompt'].apply(preprocess_text)
dtrain['answer'] = dtrain['answer'].apply(preprocess_text)

# Função para converter texto em embeddings numéricos
def text_to_tensor(text):
    """Converte texto para embeddings numéricos usando tokenização BERT."""
    tokens = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors="pt")
    return tokens.input_ids.squeeze(0)

class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        prompt = text_to_tensor(self.data.iloc[idx]['prompt'])
        answer = text_to_tensor(self.data.iloc[idx]['answer'])
        return prompt, answer

dataset = TextDataset(dtrain)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Definição do modelo neural
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instanciar modelo professor e aluno
input_dim = 50  # Dimensão da entrada
hidden_dim = 1000
output_dim = 50 #

professor_model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)
aluno_model = SimpleNN(input_dim, hidden_dim // 2, output_dim).to(device)

optimizer_aluno = optim.Adam(aluno_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Destilação do modelo professor para o aluno
for epoch in range(epochs_distillation):
    total_loss = 0
    for prompts, _ in dataloader:
        prompts = prompts.to(device, dtype=torch.float32)
        with torch.no_grad():
            professor_outputs = professor_model(prompts)
        aluno_outputs = aluno_model(prompts)
        loss = loss_fn(aluno_outputs, professor_outputs)
        optimizer_aluno.zero_grad()
        loss.backward()
        optimizer_aluno.step()
        total_loss += loss.item()
    print(f"Distillation Epoch {epoch+1}/{epochs_distillation} | Distillation Loss: {total_loss / len(dataloader):.4f}")

print("Destilação concluída.")
