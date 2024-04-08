import nltk
import torch

def caption_transform(vocab, caption, pad_word = '<pad>', start_word = '<start>', end_word='<end>', vocab_size = 64):
    token = [start_word]
    token.extend(nltk.tokenize.word_tokenize(str(caption).lower()))
    token = token[:vocab_size-1]
    token.append(end_word)
    token.extend([pad_word for i in range(vocab_size)])
    token = token[:vocab_size-1]
    result = [vocab(word) for word in token]
    return torch.Tensor(result).long()