import luigi
import pandas as pd
from task_generate_tsne_embeddings import TaskGenerateTSNEEmbeddings
from task_extract_sample import TaskExtractSample
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from bertopic import BERTopic
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

class TaskGenerateTSNEStars(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    lora_path = luigi.Parameter(default="")
    file = luigi.Parameter()

    def requires(self):
        d = dict()
        d['embeddings'] = TaskGenerateTSNEEmbeddings(domain=self.domain, sbert=self.sbert, file=self.sbert.split('/')[-1])
        d['sample'] = TaskExtractSample(domain=self.domain, is_test=True, sample_train=300_000)
        return d

    def run(self):

        df_sample = pd.read_csv(self.input()['sample'].path, delimiter='\t')
        embeddings = np.loadtxt(self.input()['embeddings'].path, delimiter="\t", dtype=np.float64)
        ratings = df_sample['rating'].tolist()
        reviews = df_sample['review'].tolist()

        
        rating_labels = []
        for rating in ratings:
            if rating >= 4:
                rating_labels.append('Positive')
            else:
                rating_labels.append('Negative')
        

        output_path = self.output().path
        output_dir = os.path.dirname(output_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        umap = UMAP(n_neighbors=15, n_components=2, metric='cosine', min_dist=0.1, random_state=42)

        reduce_embeddings = umap.fit_transform(embeddings)
        # Plot t-SNE for pos/neg/neutral
        plt.figure(figsize=(8, 6))
        colors = {'Positive': 'blue', 'Negative': 'red'}
        for rating, color in colors.items():
            indices = [i for i, label in enumerate(rating_labels) if label == rating]
            plt.scatter(reduce_embeddings[indices, 0], reduce_embeddings[indices, 1], label=rating, c=color, alpha=0.5, s=0.05)

        #plt.legend(fontsize=14)  # Increased legend font size
        plt.grid(False)
        plt.axis('off')
        # Adjust layout
        plt.tight_layout()
        # Optional: Show the plot

        plt.savefig(output_path + '/rating.png', bbox_inches="tight")
        plt.close()


        for k in [50]:
            kmeans = KMeans(n_clusters=k)
            topic_model = BERTopic(vectorizer_model=CountVectorizer(stop_words='english'), hdbscan_model=kmeans)
            print(len(reviews), embeddings.shape)
            topic_model.fit_transform(reviews, embeddings=embeddings)
            info = topic_model.get_topic_info()

            fig = topic_model.visualize_document_datamap(docs=reviews, reduced_embeddings=reduce_embeddings, interactive=False, title="", 
                                                         datamap_kwds =  {"label_font_size":18,
    "dynamic_label_size": True,
}, topics=[i for i in range(21)])
            # crop the white space
            fig.savefig(output_path + f"bertopic_{k}.png", bbox_inches="tight")
            info.to_csv(output_path + f"bertopic_{k}_freq.tsv", index=False, sep='\t')


    def output(self):
        _path = f"data/tsne_fig_stars/{self.domain}/v9/{self.file}/"
        return luigi.LocalTarget(_path)
    
if __name__ == '__main__':
    tasks = []
    for d in ['hotel', 'restaurant']:
        for a in ["all-mpnet-base-v2", f"data/sbert/{d}/all-mpnet-base-v2_lora_guided_sg_10", f"data/sbert/{d}/all-mpnet-base-v2_lora_guided_sg_250", f"data/sbert/{d}/all-mpnet-base-v2_lora_guided_senticse", f"data/sbert/{d}/all-mpnet-base-v2_lora_guided_simcse"]:
            if "StanceAware" in a:
                tasks.append(TaskGenerateTSNEStars(domain=d, sbert="all-mpnet-base-v2", lora_path=a, file=a.split('/')[-1]))
            else:
                tasks.append(TaskGenerateTSNEStars(domain=d, sbert=a, file=a.split('/')[-1]))
    luigi.build(tasks, local_scheduler=True)