import luigi
from sentence_transformers import SentenceTransformer
import json
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from task_generate_ir_acsa import TaskGenerateIRACSA
from task_train_w2v import TaskTrainW2V
from utils.SimpleEmbeddings import SimpleEmbeddings
from utils.SentiCSEmbeddings import SentiCSEmbeddings
from peft import PeftModel


class TaskEvaluateIR(luigi.Task):

    domain = luigi.ChoiceParameter(choices=['restaurant', "hotel"])
    sbert = luigi.Parameter(default="intfloat/e5-base-v2")
    lora_path = luigi.Parameter(default="")
    file = luigi.Parameter()

    def requires(self):
        d = dict()
        if self.domain == 'hotel':
            d['hotelOATS'] = TaskGenerateIRACSA(domain="hotel", folder="hotelOATS")
        else:
            d['rest16'] = TaskGenerateIRACSA(domain="restaurant", folder="rest16")
            d['rest14'] = TaskGenerateIRACSA(domain="restaurant", folder="rest14")
        if self.sbert in ['sg', 'cbow']:
            d['w2v'] = TaskTrainW2V(domain=self.domain, algo=self.sbert)
        return d
    
    def run(self):

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
                if self.lora_path == 'vahidthegreat/StanceAware-SBERT':
                    sbert[0].auto_model = PeftModel.from_pretrained(sbert[0].auto_model, "vahidthegreat/StanceAware-SBERT")
                else:
                    sbert.load_adapter(self.lora_path)
                    sbert.enable_adapters()
        
        if self.domain == 'restaurant':
            datasets = ['rest16','rest14']
        else:
            datasets = ['hotelOATS']
        
        j = defaultdict(dict)
        for d in datasets:
            with self.input()[d]['query'].open('r') as f:
                query = json.load(f)
                for idx, value in query.items():
                    query[idx] = value.lower().replace('_', ' ').strip()
                    
            with self.input()[d]['doc'].open('r') as f:
                doc = json.load(f)
            with self.input()[d]['map'].open('r') as f:
                relevant_docs = json.load(f)
            
            prompt_name = 'web_search_query' if self.sbert == 'intfloat/e5-mistral-7b-instruct' else None
            query_embs = sbert.encode(list(query.values()), show_progress_bar=True, prompt_name=prompt_name)
            docs_embs = sbert.encode(list(doc.values()), show_progress_bar=True, prompt_name=prompt_name)

            y_pred = cosine_similarity(docs_embs, query_embs)

            y_true = []
            for _ in range(len(doc)):
                y_true.append([0] * len(query_embs))

            for query_id, docs_id in relevant_docs.items():
                for d_id in docs_id:
                    y_true[int(d_id)][int(query_id)] = 1
            

            macro_map = average_precision_score(y_true, y_pred, average="macro")
            micro_map = average_precision_score(y_true, y_pred, average="micro")
            w_map = average_precision_score(y_true, y_pred, average="weighted")

            j[d]["macro_map"] = macro_map
            j[d]["micro_map"] = micro_map
            j[d]["w_map"] = w_map
       
        with self.output().open('w') as f:
            json.dump(j, f, ensure_ascii=False, indent=4)

    def output(self):
        return luigi.LocalTarget(f'data/eval_acsa/{self.domain}/{self.file}.json')

if __name__ == '__main__':
    tasks = []
    for eval in ['acsa']:
        for a in ["all-mpnet-base-v2"]:
            for d in ['restaurant', 'hotel']:
                tasks.append(TaskEvaluateIR(domain=d, sbert=a, file=a))
    luigi.build(tasks, local_scheduler=True)