import luigi
from glob import glob
import bs4
from collections import defaultdict
import ast
import csv

class TaskExtractTSNESample(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])

    def run(self):

        topics = set()

        def extract_categories(categories):
            d_topics = defaultdict(set)
            for category in categories:
                pol = category['polarity']
                if pol in ['neutral','conflict']: return set() # Not well defined
                topic = category['category'].upper()
                if topic == 'FOOD#GENERAL': return set() # mistake
                if 'MISCELLANEOUS' in topic: return set() # Not well defined
                if 'POLARITY' in topic: return set()
                e = topic.split('#')[0]
                if e == 'DRINKS': e = 'FOOD'
                if 'PRICE' in topic: continue
                d_topics[e].add(pol)
            if len(d_topics) > 1: return set()
            for key, values in d_topics.items():
                if len(values) > 1: return set()
                else : v = list(values)[0]
                return key, v
            return set()
        
        d = dict()
        folders = ['rest14', 'rest16'] if self.domain == 'restaurant' else ['hotel15', 'hotelOATS']
        for folder in folders:
            splits = ['test', 'train', 'dev']
            for split in splits:
                for file in glob(f"data/absa/{self.domain}/{folder}/{split}*"):

                    with open(file, "r") as f:
                        if folder != "hotelOATS":
                            soup = bs4.BeautifulSoup(f, features='xml')
                            sentences = soup.find_all('sentence')
                        else:
                            sentences = f.readlines()

                    
                    for sentence in sentences:
                        if folder != 'hotelOATS':
                            text = sentence.find('text').get_text().lower().strip()
                            if folder == 'rest14':
                                categories = sentence.find_all('aspectCategory')
                            else:
                                categories = sentence.find_all('Opinion')
                        else:
                            line  = sentence.split('####')
                            text = line[0].lower().strip()
                            opinions = ast.literal_eval(line[1])
                            categories = []
                            for opinion in opinions:
                                entity, aspect = opinion[1].split(' ')
                                pol = opinion[2]
                                categories.append({
                                    'category': '#'.join([entity, aspect]),
                                    'polarity': pol,
                                })

                        topics = extract_categories(categories)
                        if not len(topics): continue
                        e = topics[0]
                        e = e.replace('ROOMS_AMENITIES', 'ROOM_AMENITIES')
                        d[text] = (e.lower(), topics[1])

        datasets = []
        for text, topic in d.items():
            datasets.append([text, topic[0], topic[1]])

        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text', 'topic', 'sentiment'])
            writer.writerows(datasets)

    def output(self):
        return luigi.LocalTarget(f'data/tsne_sample/{self.domain}.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskExtractTSNESample(domain='hotel'),
        TaskExtractTSNESample(domain='restaurant')
    ]
    luigi.build(tasks, local_scheduler=True)