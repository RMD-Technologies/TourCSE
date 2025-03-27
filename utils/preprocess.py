from nltk.corpus import stopwords
from string import punctuation
from gensim.utils import simple_preprocess
import re
from sentence_transformers.models.tokenizer import WhitespaceTokenizer

stops = set(stopwords.words('english'))
stops.update(punctuation)

NUM_TOKEN = '<num>'

def preprocess(text: str) -> str:
    tokens = simple_preprocess(text, True)
    tokens = [NUM_TOKEN if re.match(r'\w*\d\w*', token) else token for token in tokens]
    tokens = [token for token in tokens if token not in stops]
    return ' '.join(tokens)