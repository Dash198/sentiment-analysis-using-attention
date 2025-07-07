# eval_all.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import os

from models import VanillaRNN, VanillaLSTM, BiRNN, BiLSTM, AttentionClassifier
from attention import AdditiveAttention, DotAttention, GeneralAttention, ConcatAttention
from data_preprocessing import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab and test set
word2idx, _, _, test_dataset = load_data()
test_loader  = DataLoader(test_dataset, batch_size=256)

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

results = []

for enc_name, enc_class in encoder_classes.items():
    for attn_name, attn_class in attention_setups.items():
        model_file = f"weights/{enc_name}_{attn_name if attn_name != 'none' else 'Base'}.pt"
        if not os.path.exists(model_file):
            print(f"Skipping {model_file} (missing)")
            continue

        print(f"\n=== Evaluating {enc_name} + {attn_name} Attention ===")

        if attn_class is None:
            model = enc_class(
                vocab_size=len(word2idx),
                embed_dim=200,
                hidden_dim=256,
                num_classes=2
            ).to(device)
        else:
            is_bi = "Bi" in enc_name
            enc = enc_class(
                vocab_size=len(word2idx),
                embed_dim=200,
                hidden_dim=256,
                num_classes=2
            ).encoder
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

        # load weights
        model.load_state_dict(torch.load(model_file, map_location=device))

        # evaluation loop
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, lengths, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                lengths = lengths.cpu()
                if attn_class is None:
                    logits = model(x, lengths)
                else:
                    logits, _ = model(x, lengths)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)

        results.append({
            "encoder": enc_name,
            "attention": attn_name,
            "accuracy": acc,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1": report['weighted avg']['f1-score'],
            "confusion_matrix": cm.tolist()
        })

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, zero_division=0))
        print(cm)

# save to CSV
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)
print("\n✅ All evaluations done, results saved to evaluation_results.csv")
