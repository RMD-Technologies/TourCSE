import luigi
from glob import glob
import bs4
from collections import defaultdict
import ast
import json

class TaskGenerateIRACSA(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    folder = luigi.Parameter()

    def run(self):

        topics = set()

        def extract_categories(categories):
            d_topics = defaultdict(set)
            topics = set()
            for category in categories:
                pol = category['polarity']
                if pol in ['neutral', 'conflict']: return set() # Not well defined
                topic = category['category'].upper()
                if topic == 'FOOD#GENERAL': return set() # mistake
                if 'MISCELLANEOUS' in topic: return set() # Not well defined
                topic = topic.replace('ROOMS_AMENITIES', 'ROOM_AMENITIES')
                e = topic.split('#')[0]
                if e in ['POLARITY', 'GENERAL', 'POSITIVE', 'NEGATIVE', 'NEUTRAL', 'HOTEL', 'RESTAURANT']: continue
                d_topics[e].add(pol)
            for key, values in d_topics.items():
                if len(values) != 1: return set()
                else : v = list(values)[0]
                topics.add((key, v))
            return topics
        
        d = dict()
        set_topics = set()
        splits = ['test', 'train']
        for split in splits:
                
            for file in glob(f"data/absa/{self.domain}/{self.folder}/{split}*"):

                with open(file, "r") as f:
                    if self.folder != "hotelOATS":
                        soup = bs4.BeautifulSoup(f, features='xml')
                        sentences = soup.find_all('sentence')
                    else:
                        sentences = f.readlines()

                
                for sentence in sentences:
                    if self.domain =='restaurant':
                        text = sentence.find('text').get_text().lower().strip()
                        if self.folder == 'rest14':
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
                    d[text] = topics
                    for topic in topics:
                        set_topics.add(topic)
        
        queries_map = {topic: id for id, topic in enumerate(set_topics)}
        documents_map = {doc: id for id, doc in enumerate(list(d.keys()))}

        relevants_doc = defaultdict(list)
        for key, values in d.items():
            for v in values:
                relevants_doc[queries_map[v]].append(str(documents_map[key]))
        
        with self.output()['query'].open('w') as f:
            q = {}
            for k, v in queries_map.items():
                entity, pol = k
                pol = 'excellent' if pol == 'positive' else 'terrible' 
                entity = entity.lower()
                q[v] = pol + ' ' + entity
            json.dump(q, f , ensure_ascii=False, indent=4)

        with self.output()['doc'].open('w') as f:
            json.dump({v:k for k, v in documents_map.items()}, f , ensure_ascii=False, indent=4)
        
        with self.output()['map'].open('w') as f:
            json.dump(relevants_doc, f , ensure_ascii=False, indent=4)


    def output(self):
        return  {
            "query": luigi.LocalTarget(f'data/ir_acsa/{self.domain}/{self.folder}_query.json'),
            "doc": luigi.LocalTarget(f'data/ir_acsa/{self.domain}/{self.folder}_doc.json'),
            "map": luigi.LocalTarget(f'data/ir_acsa/{self.domain}/{self.folder}_map.json')
        }
    
    
if __name__ == '__main__':
    tasks = [
        TaskGenerateIRACSA(domain='hotel', folder='hotelOATS'),
        TaskGenerateIRACSA(domain='restaurant', folder='rest14'),
        TaskGenerateIRACSA(domain='restaurant', folder='rest16'),
    ]
    luigi.build(tasks, local_scheduler=True)