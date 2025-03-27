import luigi
import json
import csv
import random

from task_process_cat import TaskProcessCAt


class TaskPositiveCAt(luigi.Task):

    domain = luigi.ChoiceParameter(choices=["restaurant", "hotel"])
    algo = luigi.ChoiceParameter(choices=["sg", "cbow"])
    K_topic = luigi.IntParameter()

    def requires(self):
        return TaskProcessCAt(domain=self.domain, algo=self.algo, K_topic=self.K_topic)
        
    def run(self):

        with self.input().open('r') as f:
            data = json.load(f)

        datasets = []

        def pair(el, sets):
            if len(sets) == 1: return (el, el)
            elements = [s for s in sets if s != el]
            return (el, random.choice(elements))

        for _, values in data.items():
            sets = set(values)
            for el in sets:
                t1,t2 = pair(el, sets)
                datasets.append([t1, t2, 1])
            
        
        with self.output().open('w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['text1', 'text2', 'label'])
            writer.writerows(datasets)

    def output(self):
        return luigi.LocalTarget(f'data/training/{self.domain}/cat/{self.algo}_{self.K_topic}.tsv')
    
if __name__ == '__main__':
    tasks = [
        TaskPositiveCAt(domain=domain, algo="sg", K_topic=k) for domain in ["hotel", "restaurant"] for k in [10, 50, 100, 150, 200, 250]
    ]
    luigi.build(tasks, local_scheduler=True)