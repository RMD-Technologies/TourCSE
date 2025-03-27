from reach import Reach
from tqdm import tqdm
import numpy as np

class SimpleEmbeddings():
        def __init__(self, path: str):
            self.r = Reach.load(path, unk_word="<UNK>")

        def check_vectorize(self, tokens):
            if not len(tokens): return False
            for token in tokens:
                 if token in self.r.items: return True
            return False

        def compute_pooling(self, vecs: np.array):
           return np.mean(vecs, axis=1)
         
        
        def encode(self, texts: list[str], **kwargs):
            from .preprocess import preprocess
            features = []
            for text in tqdm(texts):
                vecs = []
                tokens = preprocess(text).split()
                if not self.check_vectorize(tokens):
                    features.append(self.r['<UNK>'])
                else:
                    vecs.append(self.r.vectorize(tokens=tokens, remove_oov=True, norm=False))
                    features.append(self.compute_pooling(vecs).squeeze())
            return np.array(features)

            """
            corpus_tokens = []
            for text in tqdm(texts):
                tokens = preprocess(text).split()
                if not len(tokens): tokens = ["<UNK>"]
                corpus_tokens.append(tokens)
            features = []
            for tokens in corpus_tokens:
                if tokens == ["<UNK>"]:
                    f = np.ones(self.r['<UNK>'].shape) # Avoid zero vectors
                else:
                    f = self.r.mean_pool(tokens, remove_oov=True, safeguard=False)
                features.append(f)
            return np.array(features)
            """