# Sentiment Analysis with Attention Mechanisms

This project implements sentiment classification on the IMDB movie reviews dataset using various RNN-based models combined with different attention mechanisms. The models are trained to classify reviews as positive or negative.
Features

- Vanilla RNN, LSTM, BiRNN, BiLSTM
- Attention mechanisms (Additive, Dot, General, Concat)
- Evaluation with precision, recall, F1, and confusion matrix
- Support for loading pre-trained weights
- Command-line interface to interactively classify new reviews

## Setup

- Install dependencies

`pip install torch pandas scikit-learn matplotlib nltk`

- Download the weights
Place the pretrained .pt files in the weights/ directory (already present if you ran training).

- Run the CLI

    `python main.py`

You will be prompted to select a model, attention type, and enter a text review to classify.

## Notes

- You can retrain models using fit() methods inside each model class.

- Word-to-index vocabulary is stored as word2idx.pt.

- Evaluation metrics include accuracy, precision, recall, F1, and confusion matrix.