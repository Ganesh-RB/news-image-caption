import torch

def article_transform(tokenizer, article, max_len = 512):
    token = tokenizer.encode(article, max_length=max_len, padding="max_length",
                             truncation=True, add_special_tokens = True)
    return torch.Tensor(token).long()