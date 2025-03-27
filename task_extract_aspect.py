import luigi
from tqdm import tqdm
import nltk
import csv
from collections import Counter
from gensim.utils import simple_preprocess
from task_extract_review import TaskExtractReview
import random
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

random.seed(42)

NOUNS = {'NN', 'NNS'}

class TaskExtractAspect(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])

    def requires(self):
        return TaskExtractReview(self.domain)

    def run(self):

        c = Counter()
        with self.input().open() as f:
            
            reader = csv.DictReader(f, delimiter='\t')

            total_lines = sum(1 for _ in f) - 1
            f.seek(0) 

            for row in tqdm(reader, total=total_lines, desc="Processing rows"):

                review = row.get('review', '')
                if not review: continue
                for sent in nltk.sent_tokenize(review):
                    if len(sent.split()) > 3:
                        tokens = simple_preprocess(sent)
                        token_tags = nltk.pos_tag(tokens)
                        for token_tag in token_tags:
                            token, tag = token_tag
                            if tag in NOUNS and token not in stops:
                                c[token] += 1
        
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['aspect', 'count'])
            for aspect, count in c.most_common():
                writer.writerow([aspect, count])


    def output(self):
        return luigi.LocalTarget(f'data/aspect/{self.domain}.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskExtractAspect(domain=domain) for domain in ["hotel", 'restaurant']
    ]
    luigi.build(tasks, local_scheduler=True)