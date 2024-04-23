import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)  #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  #(batch_size,49,2048)
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

        features = last_hidden_states[0]
        return torch.Tensor(features)


#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  #(batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  #(batch_size,num_layers,attemtion_dim)

        attention_scores = self.A(combined_states)  #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)  #(batch_size,num_layers)

        alpha = F.softmax(attention_scores, dim=1)  #(batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)  #(batch_size,num_layers)

        return alpha, attention_weights


#Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, bert_encoder_dim, decoder_dim,
                 drop_prob=0.3):
        super().__init__()

        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.bert_encoder_dim = bert_encoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.para_attention = Attention(bert_encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim + bert_encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim + bert_encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim + bert_encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim + bert_encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions, articles):

        #vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features, articles)  # (batch_size, decoder_dim)

        #get the seq length to iterate
        seq_length = len(captions[0])   #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)
        para_features_num = articles.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)
        para_alphas = torch.zeros(batch_size, seq_length, para_features_num).to(device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            para_alpha, para_context = self.para_attention(articles, h)

            lstm_input = torch.cat((embeds[:, s], context, para_context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha
            para_alphas[:, s] = para_alpha

        return preds, alphas, para_alphas

    def generate_caption(self, features, articles, max_len=20, vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features, articles)  # (batch_size, decoder_dim)

        alphas = []
        para_alphas = []

        #starting input
        word = torch.tensor(vocab.word2idx['<start>']).view(1, -1).to(device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)
            para_alpha, para_context = self.para_attention(articles, h)

            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            para_alphas.append(para_alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context, para_context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)

            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated word
            captions.append(predicted_word_idx.item())

            #end if <EOS detected>
            if vocab.idx2word[predicted_word_idx.item()] == "<end>":
                break

            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        #covert the vocab idx to words and return sentence
        return [vocab.idx2word[idx] for idx in captions], alphas, para_alphas

    def init_hidden_state(self, encoder_out, bert_encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        mean_bert_encoder_out = bert_encoder_out.mean(dim=1)
        inp = torch.cat((mean_encoder_out, mean_bert_encoder_out), 1)  #TODO: check for cat dim argument severty:low
        h = self.init_h(inp)  # (batch_size, decoder_dim)
        c = self.init_c(inp)
        return h, c

    def sample(self, features, articles, max_len=20):
                # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features, articles)  # (batch_size, decoder_dim)

        alphas = []
        para_alphas = []

        #starting input
        word = torch.tensor(vocab.word2idx['<start>']).view(1, -1).to(device)
        embeds = self.embedding(word)

        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, h)
            para_alpha, para_context = self.para_attention(articles, h)

            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())
            para_alphas.append(para_alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context, para_context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size, -1)
            
        return captions, alphas, para_alphas

class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, bert_encoder_dim, decoder_dim, drop_prob=0.3,
                 bert_model=None):
        super().__init__()
        self.encoder = EncoderCNN()
        self.bert_encoder = EncoderBert(embed_size, bert_model)
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            bert_encoder_dim=bert_encoder_dim,
            decoder_dim=decoder_dim,
            drop_prob=drop_prob
        )

    def forward(self, images, captions, articles):
        features = self.encoder(images)
        bert_out = self.bert_encoder(articles)
        outputs = self.decoder(features, captions, bert_out)
        return outputs