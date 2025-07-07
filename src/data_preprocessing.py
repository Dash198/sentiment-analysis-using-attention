import torch
from torch.nn.utils.rnn import pad_sequence
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

from collections import Counter

import os

weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
word2idx_path = os.path.join(weights_dir, "word2idx.pt")
word2idx = torch.load(word2idx_path)


stop_words = set(stopwords.words('english'))

class TextDataset(torch.utils.data.Dataset):
    # Class tailored for our data.

    def __init__(self, padded_seqs, lengths, labels):
        self.padded_seqs = padded_seqs  # [N, max_len]
        self.lengths = lengths          # [N]
        self.labels = labels            # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.padded_seqs[idx],     # tensor [max_len]
            self.lengths[idx],        # scalar
            self.labels[idx]           # scalar
        )

def load_data(train_frac=0.8, test_frac=0.1, val_frac=0.1):
    splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet', 'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["test"])
    df = pd.concat([df_train, df_test])

    tokenized_text = [clean_and_tokenize_text(text) for text in df['text']]
    word2idx = build_vocab(tokenized_text, min_freq=1)

    df['encoded_text'] = [
        tokens_to_indices(tokens, word2idx)
        for tokens in tokenized_text
    ]

    train_df, temp_df = train_test_split(
        df,
        test_size=test_frac + val_frac,
        stratify=df['label'],   # preserve positive/negative balance if labeled
        random_state=42
    )

    # Then split val and test from that 20%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_frac/test_frac+val_frac),
        stratify=temp_df['label'],
        random_state=42
    )

    def split_df(df):
        padded_seqs, lengths = pad_seqs(df['encoded_text'])
        labels = torch.tensor(df['label'].values)
        return TextDataset(padded_seqs, lengths, labels)

    train_dataset = split_df(train_df)
    val_dataset = split_df(val_df)
    test_dataset = split_df(test_df)

    return word2idx, train_dataset, val_dataset, test_dataset

def clean_and_tokenize_text(text):
    # Helper function which removes unnecessary characters and splits the text into tokens
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

def build_vocab(token_lists, min_freq=1):

    # Makes a vocabulary indexing of the tokens
    counter = Counter(token for tokens in token_lists for token in tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab

def tokens_to_indices(tokens, vocab):
    # Uses said indexing to index each token list
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_seqs(encoded_seqs):
  # Function that pads up the shorter sequences

  tensor_seqs = [torch.tensor(seq) for seq in encoded_seqs]
  padded = pad_sequence(tensor_seqs, batch_first=True, padding_value = word2idx['<PAD>'])
  lengths = torch.tensor([len(seq) for seq in tensor_seqs])

  return padded, lengths

def encode_review(text, word2idx):
    tokens = clean_and_tokenize_text(text)
    indices = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    return torch.tensor(indices).unsqueeze(0)
