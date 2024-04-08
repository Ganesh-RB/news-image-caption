from collections import Counter
import os.path
import pickle
import json
import nltk

# uncomment this line, if get error Resource punkt not found. Please use the NLTK Downloader to obtain the resource:
nltk.download('punkt')


class Vocabulary(object):
    """Vocabulary for a dataset.
    Note: Vocabulary is generated from the captions only.
    """

    def __init__(self, captionFile, vocab_threshold, vocab_file='vocab.pkl', start_word="<start>", end_word="<end>",
                 unk_word="<unk>"):
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.captionFile = captionFile

        if os.path.exists(self.vocab_file):
            print("loading already saved file")
            with open(self.vocab_file, 'rb') as f:
                pretrained_vocab = pickle.load(f)
                self.word2idx = pretrained_vocab.word2idx
                self.idx2word = pretrained_vocab.idx2word
        else:
            self.init_vocab()
            self.build_vocab()

    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}

        # Add the special tokens to the vocabulary.
        self.word2idx[self.start_word] = 0
        self.word2idx[self.end_word] = 1
        self.word2idx[self.unk_word] = 2

        self.idx2word[0] = self.start_word
        self.idx2word[1] = self.end_word
        self.idx2word[2] = self.unk_word

    def build_vocab(self):
        # Build the vocabulary.
        # coco = COCO(self.annFile)
        # ids = coco.anns.keys()
        counter = Counter()
        json_data = json.load(open(self.captionFile))

        i = 0
        for key, item in json_data.items():
            for idx, caption in item['images'].items():
                i += 1
                tokens = nltk.tokenize.word_tokenize(caption.lower())
                counter.update(tokens)

                if i % 100000 == 0:
                    print("[{}] Tokenized the captions.".format(i))

        # If the word frequency is less than 'vocab_threshold', then the word is discarded.
        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        # Create a mapping from the words to the indices.
        idx = 3
        for word in words:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        # Save the vocabulary to a file.
        with open(self.vocab_file, 'wb') as f:
            pickle.dump(self, f)

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


vocab = Vocabulary("data/caption.json", 5,
                   "/content/drive/MyDrive/Academic/Sem8/NLP/Project/news-image-caption/models/vocab.pkl", )
