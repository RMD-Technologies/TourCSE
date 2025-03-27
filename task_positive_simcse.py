import luigi
import csv
import pandas as pd
import nltk

from task_extract_sample import TaskExtractSample

class TaskPositiveSimCSE(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])

    def requires(self):
        return TaskExtractSample(self.domain)

    def run(self):

        df = pd.read_csv(self.input().path, delimiter='\t')

        df = df[df['split'] == 'train']

        reviews = df['review'].tolist()

        datasets = []
        for r in reviews:
            for sent in nltk.sent_tokenize(r):
                if len(sent.split()) < 3: continue
                datasets.append([sent, sent, 1])
        
                
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(datasets)

    def output(self):
        return luigi.LocalTarget(f'data/training/{self.domain}/simcse.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskPositiveSimCSE(domain=domain) for domain in ['restaurant', 'hotel']
    ]
    luigi.build(tasks, local_scheduler=True)