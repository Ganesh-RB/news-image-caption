import numpy as np
import torch
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
        padded = articles
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
