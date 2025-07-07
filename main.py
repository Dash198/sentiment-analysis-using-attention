import torch
import pickle
from src.models import VanillaRNN, VanillaLSTM, BiRNN, BiLSTM, AttentionClassifier
from src.attention import AdditiveAttention, DotAttention, GeneralAttention, ConcatAttention
from src.data_preprocessing import clean_and_tokenize_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vocab
word2idx = torch.load('./weights/word2idx.pt')

model_classes = {
    "VanillaRNN": VanillaRNN,
    "VanillaLSTM": VanillaLSTM,
    "BiRNN": BiRNN,
    "BiLSTM": BiLSTM
}

attention_classes = {
    "Additive": AdditiveAttention,
    "Dot": DotAttention,
    "General": GeneralAttention,
    "Concat": ConcatAttention
}

def classify(review, model_choice, attention_choice):
    tokens = clean_and_tokenize_text(review)
    indices = [word2idx.get(tok, word2idx["<UNK>"]) for tok in tokens]
    tensor = torch.tensor(indices).unsqueeze(0).to(device)
    length = torch.tensor([len(indices)]).cpu()

    if attention_choice == "None":
        ModelClass = model_classes[model_choice]
        model = ModelClass(vocab_size=len(word2idx), embed_dim=200, hidden_dim=256, num_classes=2)
        model.load_state_dict(torch.load(f"./weights/{model_choice}.pt"))
    else:
        enc = model_classes[model_choice](vocab_size=len(word2idx), embed_dim=200, hidden_dim=256, num_classes=2).rnn
        is_bi = getattr(enc, 'bidirectional', False)
        enc_dim = 256 * (2 if is_bi else 1)
        attention = attention_classes[attention_choice](enc_dim=enc_dim, dec_dim=enc_dim, att_dim=128)
        model = AttentionClassifier(
            vocab_size=len(word2idx),
            embed_dim=200,
            hidden_dim=256,
            num_classes=2,
            encoder_rnn=enc,
            attention_mech=attention
        )
        model.load_state_dict(torch.load(f"./weights/{model_choice}_{attention_choice}Attention.pt"))

    model.to(device)
    model.eval()

    with torch.no_grad():
        if attention_choice == "None":
            logits = model(tensor, length)
        else:
            logits, _ = model(tensor, length)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    labels = ["negative", "positive"]
    return labels[pred], float(probs[0][pred])

if __name__ == "__main__":
    print("Available models: ", list(model_classes.keys()))
    model_choice = input("Choose a model: ")

    print("Available attentions: None, Additive, Dot, General, Concat")
    attention_choice = input("Choose attention (or None): ")

    while True:
        review = input("\nEnter your review (or type 'quit' to exit):\n> ")
        if review.lower() == "quit":
            break
        sentiment, prob = classify(review, model_choice, attention_choice)
        print(f"\nPrediction: {sentiment} ({prob:.2%} confidence)\n")
