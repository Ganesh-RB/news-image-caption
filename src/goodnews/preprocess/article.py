import torch
import numpy as np

def article_transform(tokenizer, model, article, max_len = 2048):
    token = tokenizer.encode(article, add_special_tokens=False)
    padded = [101]
    padded.extend(token)
    padded = padded[:max_len-1]
    padded.extend([102])
    padded.extend([0]*(max_len))
    padded[:max_len],tokenizer.decode(padded[:max_len])
