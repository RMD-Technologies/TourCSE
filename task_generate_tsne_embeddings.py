import luigi
import pandas as pd
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from task_extract_sample import TaskExtractSample
from task_train_w2v import TaskTrainW2V
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from bertopic import BERTopic
from utils.SimpleEmbeddings import SimpleEmbeddings
from utils.SentiCSEmbeddings import SentiCSEmbeddings
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


class TaskGenerateTSNEEmbeddings(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    lora_path = luigi.Parameter(default="")
    file = luigi.Parameter()

    def requires(self):
        d = dict()
        d['sample'] = TaskExtractSample(domain=self.domain, is_test=True, sample_train=300_000) 
        if self.sbert in ['sg', 'cbow', 'sg_hs', 'cbow_hs']:
            d['w2v'] = TaskTrainW2V(domain=self.domain, algo=self.sbert)
        return d

    def run(self):

        df = pd.read_csv(self.input()['sample'].path, delimiter='\t')


        if self.sbert in ['sg', 'cbow']:
            sbert = SimpleEmbeddings(self.input()['w2v'].path)
        elif 'SentiCSE' in self.sbert:
            sbert = SentiCSEmbeddings(self.sbert)
        elif 'potion' in self.sbert:
            from sentence_transformers.models import StaticEmbedding
            static_embedding = StaticEmbedding.from_model2vec(self.sbert)
            sbert = SentenceTransformer(modules=[static_embedding])
        else:
            sbert = SentenceTransformer(self.sbert, trust_remote_code=True)

            if self.lora_path:
                sbert.load_adapter(self.lora_path)
                sbert.enable_adapters()

        reviews = df['review'].tolist()

        embeddings = sbert.encode(reviews, show_progress_bar=True)

        output_path = self.output().path
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Convert to Pandas DataFrame
        np.savetxt(self.output().path, embeddings, delimiter="\t", fmt="%.18g")  # Use "%.3f" for floats
       

    def output(self):
        _path = f"data/embeddings/{self.domain}/{self.file}.tsv"
        return luigi.LocalTarget(_path)
    
if __name__ == '__main__':
    tasks = []
    for d in ['restaurant', 'hotel']:
        for a in ["cbow", "sg","all-mpnet-base-v2", f"data/sbert/{d}/all-mpnet-base-v2_lora_guided_sg_10", f"data/sbert/{d}/all-mpnet-base-v2_lora_guided_sg_250"]:
            tasks.append(TaskGenerateTSNEEmbeddings(domain=d, sbert=a, file=a.split('/')[-1]))
    luigi.build(tasks, local_scheduler=True)