import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics import accuracy_score, classification_report

import os

weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
word2idx_path = os.path.join(weights_dir, "word2idx.pt")
word2idx = torch.load(word2idx_path)


class VanillaRNN(nn.Module):

  def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, num_classes)

  def forward(self, x, lengths):
    embed = self.embedding(x)

    packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
    _, hidden = self.rnn(packed)

    logits = self.fc(hidden.squeeze(0))
    return logits

  def fit(self, train_loader, val_loader=None, epochs=5, lr=1e-3, device='cpu', stopping_criterion=1e-2):
    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    self.to(device)
    self.train()

    for epoch in range(epochs):
      self.train()
      total_loss = 0
      for x, lengths, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        optimizer.zero_grad()
        logits = self(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}')

      if val_loader:
        self.evaluate(val_loader, device)

  def evaluate(self, val_loader, device,report=False):
    self.eval()
    with torch.no_grad():
      preds = torch.tensor([], dtype=torch.long)
      true = torch.tensor([], dtype=torch.long)
      for x, lengths, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        logits = self(x,lengths)
        preds_batch = torch.argmax(logits, dim=1)
        preds = torch.cat([preds, preds_batch.cpu()])
        true = torch.cat([true, y.cpu()])

      if report:
        print(classification_report(true, preds))
      else:
        print(f'Validation Accuracy: {accuracy_score(true, preds):.4f}')

class BiRNN(nn.Module):

  def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.fc = nn.Linear(2 * hidden_dim, num_classes)

  def forward(self, x, lengths):
    embed = self.embedding(x)

    packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
    _, hidden = self.rnn(packed)
    hidden_cat = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
    logits = self.fc(hidden_cat)
    return logits

  def fit(self, train_loader, val_loader=None, epochs=5, lr=1e-3, device='cpu', stopping_criterion=1e-2):
    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    self.to(device)
    self.train()

    for epoch in range(epochs):
      self.train()
      total_loss = 0
      for x, lengths, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        optimizer.zero_grad()
        logits = self(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}')

      if val_loader:
        self.evaluate(val_loader, device)

  def evaluate(self, val_loader, device,report=False):
    self.eval()
    with torch.no_grad():
      preds = torch.tensor([], dtype=torch.long)
      true = torch.tensor([], dtype=torch.long)
      for x, lengths, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        logits = self(x,lengths)
        preds_batch = torch.argmax(logits, dim=1)
        preds = torch.cat([preds, preds_batch.cpu()])
        true = torch.cat([true, y.cpu()])

      if report:
        print(classification_report(true, preds))
      else:
        print(f'Validation Accuracy: {accuracy_score(true, preds):.4f}')

class VanillaLSTM(nn.Module):
  def __init__(self, vocab_size, hidden_dim, embed_dim, num_classes):
     super().__init__()
     self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=word2idx['<PAD>'])
     self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
     self.fc = nn.Linear(hidden_dim, num_classes)

  def forward(self, x, lengths):
    embed = self.embedding(x)

    packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
    _, (hidden,cell) = self.rnn(packed)

    logits = self.fc(hidden.squeeze(0))
    return logits

  def fit(self, train_loader, val_loader=None, epochs=5, lr=1e-3, device='cpu', stopping_criterion=1e-2):
    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    self.to(device)
    self.train()

    for epoch in range(epochs):
      self.train()
      total_loss = 0
      for x, lengths, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        optimizer.zero_grad()
        logits = self(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}')

      if val_loader:
        self.evaluate(val_loader, device)

  def evaluate(self, val_loader, device,report=False):
    self.eval()
    with torch.no_grad():
      preds = torch.tensor([], dtype=torch.long)
      true = torch.tensor([], dtype=torch.long)
      for x, lengths, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        logits = self(x,lengths)
        preds_batch = torch.argmax(logits, dim=1)
        preds = torch.cat([preds, preds_batch.cpu()])
        true = torch.cat([true, y.cpu()])

      if report:
        print(classification_report(true, preds))
      else:
        print(f'Validation Accuracy: {accuracy_score(true, preds):.4f}')

class BiLSTM(nn.Module):
  def __init__(self, vocab_size, hidden_dim, embed_dim, num_classes):
     super().__init__()
     self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim, padding_idx=word2idx['<PAD>'])
     self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
     self.fc = nn.Linear(2*hidden_dim, num_classes)

  def forward(self, x, lengths):
    embed = self.embedding(x)

    packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
    _, (hidden,cell) = self.rnn(packed)
    hidden_cat = torch.cat((hidden[0,:,:], hidden[1,:,:]), dim=1)
    logits = self.fc(hidden_cat)
    return logits

  def fit(self, train_loader, val_loader=None, epochs=5, lr=1e-3, device='cpu', stopping_criterion=1e-2):
    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    self.to(device)
    self.train()

    for epoch in range(epochs):
      self.train()
      total_loss = 0
      for x, lengths, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        optimizer.zero_grad()
        logits = self(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      print(f'Epoch {epoch+1}, Training Loss: {avg_loss:.4f}')

      if val_loader:
        self.evaluate(val_loader, device)

  def evaluate(self, val_loader, device,report=False):
    self.eval()
    with torch.no_grad():
      preds = torch.tensor([], dtype=torch.long)
      true = torch.tensor([], dtype=torch.long)
      for x, lengths, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        lengths = lengths.cpu()

        logits = self(x,lengths)
        preds_batch = torch.argmax(logits, dim=1)
        preds = torch.cat([preds, preds_batch.cpu()])
        true = torch.cat([true, y.cpu()])

      if report:
        print(classification_report(true, preds))
      else:
        print(f'Validation Accuracy: {accuracy_score(true, preds):.4f}')

class AttentionClassifier(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, encoder_rnn, attention_mech):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<PAD>'])
    self.encoder = encoder_rnn
    self.attention = attention_mech
    self.fc = nn.Linear(hidden_dim * (2 if getattr(self.encoder, 'bidirectional', False) else 1), num_classes)

  def forward(self, x, lengths):
    embed = self.embedding(x)
    packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)

    if isinstance(self.encoder, nn.LSTM):
      enc_outs, (h,_) = self.encoder(packed)
    else:
      enc_outs, h = self.encoder(packed)

    enc_outs, _ = pad_packed_sequence(enc_outs, batch_first=True)
    enc_outs = enc_outs.transpose(0,1)

    if getattr(self.encoder, 'bidirectional', False):
      query = torch.cat((h[-2], h[-1]), dim=1)
    else:
      query = h[-1]

    ctx, alpha = self.attention(query, enc_outs)
    logits = self.fc(ctx)

    return logits, alpha

  def fit(self, train_loader, val_loader=None, epochs=5, lr=1e-3, device='cpu'):
    optimizer = torch.optim.Adam(self.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    self.to(device)

    for epoch in range(epochs):
        self.train()
        total_loss = 0
        for x, lengths, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            lengths = lengths.cpu()

            optimizer.zero_grad()
            logits, _ = self(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

        if val_loader:
            self.evaluate(val_loader, device)


  def evaluate(self, val_loader, device, report=False):
      self.eval()

      all_preds = []
      all_true  = []

      with torch.no_grad():
          for x, lengths, y in val_loader:
              x = x.to(device)
              y = y.to(device)
              lengths = lengths.cpu()

              logits, _ = self(x, lengths)
              preds = torch.argmax(logits, dim=1)

              all_preds.extend(preds.cpu().tolist())
              all_true.extend(y.cpu().tolist())

      if report:
          print(classification_report(all_true, all_preds))
      else:
          acc = accuracy_score(all_true, all_preds)
          print(f"Validation Accuracy: {acc:.4f}")