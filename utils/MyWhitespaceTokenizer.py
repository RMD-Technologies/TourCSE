from sentence_transformers.models.tokenizer import WhitespaceTokenizer
from .preprocess import preprocess

class MyWhitespaceTokenizer(WhitespaceTokenizer):

    def tokenize(self, text, **kwargs):
        tokens = preprocess(text)
        return super().tokenize(tokens, **kwargs)