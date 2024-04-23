import numpy as np
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features


class EncoderBert(nn.Module):
    def __init__(self, embed_size, bert_model):
        super(EncoderBert, self).__init__()
        self.bert = bert_model

    def forward(self, articles):
        padded = articles.cpu().data.numpy()
        attention_mask = np.where(padded != 0, 1, 0)
        # Getting vectors
        input_ids = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)

        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_states['last_hidden_state'].size()

        features = last_hidden_states[0][:, 0, :].numpy()
        return torch.Tensor(features)


class Encoder(nn.Module):
    def __init__(self, embed_size, bert_model):
        super(Encoder, self).__init__()
        self.image_encoder = EncoderCNN(embed_size)
        self.article_encoder = EncoderBert(embed_size, bert_model)

    def forward(self, images, articles):
        image_features = self.image_encoder(images)
        article_features = self.article_encoder(articles)
        return torch.cat((image_features, article_features), 1)


import torch
import torch.nn as nn
from torch.nn import functional as F


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hidden, _ = self.lstm(inputs)
        outputs = self.linear(hidden)
        return outputs, None, None

    def sample(self, inputs, states=None, max_len=20):
        """Accept a pre-processed image tensor (inputs) and return predicted
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        """
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        """Accept a pre-processed image tensor and return the top predicted
        sentences. This is the beam search approach.
        """
        # Top word idx sequences and their corresponding inputs and states
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform outputs to log probabilities to avoid floating-point
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]



class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, bert_encoder_size=768, decoder_dim=512, bert_model=None):
        super().__init__()
        self.encoder = Encoder(embed_size, bert_model)
        self.decoder = DecoderRNN(embed_size + bert_encoder_size, decoder_dim, vocab_size)

    def forward(self, images, captions, articles):
        features = self.encoder(images, articles)
        outputs = self.decoder(features, captions)
        return outputs