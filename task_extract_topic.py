import luigi
import csv
from collections import Counter
import json
from reach import Reach
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from task_extract_aspect import TaskExtractAspect
from task_train_w2v import TaskTrainW2V

class TaskExtractTopic(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    algo = luigi.ChoiceParameter(choices=['sg', 'cbow'])
    K_topic = luigi.IntParameter()
    N_ASPECT = luigi.IntParameter(default=1000)


    def requires(self):
        return [
            TaskTrainW2V(self.domain, self.algo),
            TaskExtractAspect(self.domain)
        ]

    def run(self):

        r = Reach.load(self.input()[0].path)

        c = Counter()
        with self.input()[1].open('r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                c[row['aspect']] = int(row['count'])
        
        aspects = []
        vecs = []
        for aspect, _ in c.most_common(self.N_ASPECT):
            if aspect in r.items:
                aspects.append(aspect)
                vecs.append(r[aspect])

        vecs = np.array(vecs)
        cosine_distance = pairwise_distances(vecs, metric='cosine')
        condensed_distance = squareform(cosine_distance)
        Z = linkage(condensed_distance, method='ward')
        labels = fcluster(Z, self.K_topic, criterion='maxclust')
   
        clustered_aspects = {}

        # Iterate over the cluster labels and aspects
        for label, aspect in zip(labels, aspects):
            if label not in clustered_aspects:
                clustered_aspects[label] = []
            clustered_aspects[label].append(aspect)

        # Create a new dictionary with label names replaced by the first aspect (upper case)
        final_clustered_aspects = {}

        for label, aspects in clustered_aspects.items():
            # Get the first aspect in the cluster, convert it to uppercase, and use it as the new label
            new_label = aspects[0].upper()
            final_clustered_aspects[new_label] = aspects

        # Save the dictionary to a JSON file
        with self.output().open('w') as f:
            json.dump(final_clustered_aspects, f, indent=4)


    def output(self):
       return luigi.LocalTarget(f'data/topic/{self.domain}/{self.algo}_{self.K_topic}.json')
        
    
if __name__ == '__main__':
    tasks = [
        TaskExtractTopic(domain=domain, algo=algo, K_topic=k) for domain in ["restaurant"] for algo in ['sg'] for k in [10, 250]
    ]
    luigi.build(tasks, local_scheduler=True)