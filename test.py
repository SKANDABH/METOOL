import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import string
import nltk

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

with open("math.txt", "r",encoding='utf-8') as al:
    data = al.read()


def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = word_tokenize(text)
    return tokens


cleaned_data = clean_text(data)

vocab = set(cleaned_data)
vocab.add(' ')  # Add space to vocabulary
vocab_size = len(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for i, word in enumerate(vocab)}

with open("vocab.txt", "w",encoding='utf-8') as f:
    for word in vocab:
        f.write(word + "\n")
text_as_int = [word_to_index.get(word, word_to_index[' ']) for word in cleaned_data]    # Handle unknown words

seq_length = 100
input_texts = []
output_texts = []

for i in range(len(text_as_int) - seq_length):
    input_texts.append(text_as_int[i:i + seq_length])
    output_texts.append(text_as_int[i + seq_length])

X = torch.tensor(input_texts, dtype=torch.long)
y = torch.tensor(output_texts, dtype=torch.long)


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 64
dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seq_length):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


embed_size = 256
hidden_size = 256
model = LSTMModel(vocab_size, embed_size, hidden_size, seq_length)

state_dict = torch.load('modeltorchmath45.pth', map_location=torch.device('cpu'))

current_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_state_dict and v.size() == current_state_dict[k].size()}
model.load_state_dict(filtered_state_dict, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



def generate_text(model, start_string, length=4, temperature=1.0):
    model.eval()
    cleaned_start_string = clean_text(start_string)
    input_indices = [word_to_index.get(word, word_to_index[' ']) for word in cleaned_start_string]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_string

    with torch.no_grad():
        for _ in range(length):
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output / temperature, dim=-1)
            predicted_index = torch.multinomial(probabilities, 1).item()

            predicted_word = index_to_word.get(predicted_index, ' ')
            generated_text += " " + predicted_word

            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_index]], device=device)), dim=1)

    return generated_text
start_string = "Calculate the volume of a cone with a slant"
generated_text = generate_text(model, start_string, length=10)
print(generated_text)