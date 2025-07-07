import torch
from torch.utils.data import DataLoader
from models import VanillaRNN, VanillaLSTM, BiRNN, BiLSTM, AttentionClassifier
from attention import AdditiveAttention, DotAttention, GeneralAttention, ConcatAttention
from data_preprocessing import load_data
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab and dataset
word2idx, train_dataset, val_dataset, test_dataset = load_data()
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=256)
test_loader  = DataLoader(test_dataset, batch_size=256)

# create weights directory if missing
os.makedirs("weights", exist_ok=True)

# define encoders
encoder_classes = {
    "VanillaRNN": VanillaRNN,
    "VanillaLSTM": VanillaLSTM,
    "BiRNN": BiRNN,
    "BiLSTM": BiLSTM
}

# define attentions
attention_setups = {
    "none": None,
    "Additive": AdditiveAttention,
    "Dot": DotAttention,
    "General": GeneralAttention,
    "Concat": ConcatAttention
}

for enc_name, enc_class in encoder_classes.items():
    for attn_name, attn_class in attention_setups.items():
        print(f"\n=== Training {enc_name} + {attn_name} Attention ===\n")

        if attn_class is None:
            # Vanilla models
            model = enc_class(
                vocab_size=len(word2idx),
                embed_dim=200,
                hidden_dim=256,
                num_classes=2
            ).to(device)
        else:
            # AttentionClassifier
            # adapt hidden dimension
            is_bi = "Bi" in enc_name
            enc = enc_class(
                vocab_size=len(word2idx),
                embed_dim=200,
                hidden_dim=256,
                num_classes=2
            ).encoder # reuse the RNN/LSTM module
            enc_dim = 256 * (2 if is_bi else 1)

            attn = attn_class(enc_dim, 256, 128) if attn_name in ["Additive", "Concat"] else attn_class(enc_dim, 256)

            model = AttentionClassifier(
                vocab_size=len(word2idx),
                embed_dim=200,
                hidden_dim=256,
                num_classes=2,
                encoder_rnn=enc,
                attention_mech=attn
            ).to(device)

        try:
            model.fit(train_loader, val_loader, device=device, epochs=10)
            model.evaluate(test_loader, device=device, report=True)

            torch.save(
                model.state_dict(),
                f"weights/{enc_name}_{attn_name if attn_name != 'none' else 'Base'}.pt"
            )
        except Exception as e:
            print(f"Error training {enc_name} + {attn_name} Attention: {e}")
