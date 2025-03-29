import luigi
import csv
import pandas as pd
from collections import defaultdict
import nltk
import random

from task_extract_sample import TaskExtractSample

class TaskPositiveSentiCSE(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])

    def requires(self):
        return TaskExtractSample(self.domain)

    def run(self):

        df = pd.read_csv(self.input().path, delimiter='\t')

        d = defaultdict(list)

        reviews = df['review'].tolist()
        ratings = df['rating'].tolist()

        for rating, review in zip(ratings, reviews):
            for sent in nltk.sent_tokenize(review):
                if len(sent.split()) < 3: continue
                d[rating].append(sent)
        

        datasets = []
        for _, values in d.items():
            for v in values:
                datasets.append(
                    [v , random.choice(values), 1]
                )
                
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(datasets)

    def output(self):
        return luigi.LocalTarget(f'data/training/{self.domain}/senticse.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskPositiveSentiCSE(domain=domain) for domain in ['restaurant', 'hotel']
    ]
    luigi.build(tasks, local_scheduler=True)