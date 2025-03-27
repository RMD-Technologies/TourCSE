import luigi
from tqdm import tqdm
import nltk
import csv

from task_extract_review import TaskExtractReview
from utils.preprocess import preprocess

class TaskPreprocessW2V(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])

    def requires(self):
        return TaskExtractReview(self.domain)

    def run(self):

        sentences = []

        with self.input().open() as f:
            
            reader = csv.DictReader(f, delimiter='\t')

            total_lines = sum(1 for _ in f) - 1
            f.seek(0) 

            for row in tqdm(reader, total=total_lines, desc=f"Processing {self.domain}"):

                title = row.get('title', '')
                review = row.get('review', '')
                r = '. '.join([title, review])
                for sent in nltk.sent_tokenize(r):
                    if len(sent.split()) > 3:
                        sent_clean = preprocess(sent)
                        if len(sent_clean.split()) > 3:
                            sentences.append([sent_clean])

        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(sentences)

    def output(self):
        return luigi.LocalTarget(f'data/preprocess/{self.domain}.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskPreprocessW2V(domain=domain) for domain in ["hotel", 'restaurant']
    ]
    luigi.build(tasks, local_scheduler=True)