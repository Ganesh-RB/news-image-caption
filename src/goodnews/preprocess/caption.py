import torch
import spacy
import nltk
import re


def preprocess_caption(caption):
    """Preprocess caption text by removing punctuations and making lowercase."""
    caption = re.sub(r'[^\w\s]', '', caption)
    caption = caption.lower()
    return caption


nlp = spacy.load('en_core_web_sm')


def replace_entities(text):
    """Replace entities in text with their labels (ORG, PERSON, GPE, DATE)."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PERSON', 'GPE', 'DATE']:
            text = text.replace(ent.text, '<' + ent.label_ + '>')
    return text


def caption_transform(vocab, caption, pad_word='<pad>', start_word='<start>', end_word='<end>', vocab_size=64):
    """Transform caption text into a tensor of word indices."""

    caption = preprocess_caption(caption)
    caption = replace_entities(caption)

    token = [start_word]
    token.extend(nltk.tokenize.word_tokenize(str(caption).lower()))
    token = token[:vocab_size - 1]
    token.append(end_word)
    token.extend([pad_word for _ in range(vocab_size)])
    token = token[:vocab_size - 1]
    result = [vocab(word) for word in token]
    return torch.Tensor(result).long()
