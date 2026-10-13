"""Frozen full-corpus WASM comparison; no aliases or test-driven selection."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from bilingual import retrieval_metrics, retrieval_details

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/pilot-fa-en-domain'


def fixtures(challenge=None):
    paths=[ROOT/'eval/quality/corpus/persian-domain-hard.json',ROOT/'eval/quality/corpus/persian-native-demos.json',
           ROOT/'training/distill/cache/mini-fa-en-round3-native/qa-test.json']
    paths += [ROOT/f'eval/quality/corpus/bilingual-hard-round{i}.json' for i in [2,3,4]]
    fresh,previous,qa,*old=[json.loads(p.read_text()) for p in paths]
    result={**fresh['suites'],'native_qa':qa}
    for task,corpus in previous['corpora'].items():
        result['previous_'+task]=dict(corpus=corpus,queries=previous['queries'][task])
    for i,f in zip([2,3,4],old):
        for lang in ['fa','en']:
            result[f'round{i}_{lang}']={k:[r for r in f[k] if r['lang']==lang] for k in ['queries','corpus']}
    reasoning=ROOT/'eval/quality/corpus/persian-reasoning-hard.json'
    result.update({'reasoning_'+k:f for k,f in json.loads(reasoning.read_text())['suites'].items()})
    paths.append(reasoning)
    if challenge is not None:
        extra=json.loads(challenge.read_text())
        result.update({'fresh_'+k:f for k,f in extra['suites'].items()});paths.append(challenge)
    return result,{str(p.resolve().relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def score_fixture(vectors,lookup,f):
    docs=f['corpus'];queries=f['queries'];ids={d['id']:i for i,d in enumerate(docs)}
    scores=vectors[[lookup[q['text']] for q in queries]]@vectors[[lookup[d['text']] for d in docs]].T
    rel=[{ids[k]:grade for k,grade in q['relevant'].items()} for q in queries]
    result=retrieval_metrics(scores,rel)
    details=retrieval_details(scores,queries,docs)
    result.update(count=len(queries),corpus_count=len(docs),top1=sum(q['top1_correct'] for q in details['queries'])/len(queries),**details)
    for i,q in enumerate(result['queries']):
        q['ndcg']=retrieval_metrics(scores[i:i+1],[rel[i]])['ndcg@10']
        q['top_text']=docs[ids[q['top1']]]['text']
    paired={}
    for q,d in zip(queries,details['queries']):
        if 'pair' in q:paired.setdefault((q['category'],q['pair']),[]).append(d)
    result['pairs']={}
    for category in {k[0] for k in paired}:
        groups=[v for k,v in paired.items() if k[0]==category]
        assert all(len(g)==2 for g in groups)
        result['pairs'][category]=dict(count=len(groups),both_contrasts_correct=sum(all(q['contrast_correct'] for q in g) for g in groups),
                                      both_top1_correct=sum(all(q['top1_correct'] for q in g) for g in groups))
    return result


def evaluate(name,glue,output=OUT,challenge=None):
    output.mkdir(parents=True,exist_ok=True)
    destination=output/f'eval-{name}.json'
    if destination.exists():raise FileExistsError('Preserve consumed tests')
    cases,inputs=fixtures(challenge);texts=list(dict.fromkeys(r['text'] for f in cases.values() for r in f['corpus']+f['queries']))
    raw=subprocess.run(['node',str(ROOT/'eval/quality/collect-wasm.mjs'),str(glue)],input=json.dumps(texts),text=True,capture_output=True,check=True)
    data=json.loads(raw.stdout);v=np.asarray(data['embeddings'],dtype=np.float32)
    assert v.shape==(len(texts),384) and np.isfinite(v).all()
    lookup={t:i for i,t in enumerate(texts)}
    result={k:data[k] for k in ['model_sha256','tokenizer_sha256','wasm_sha256','wasm_bytes','gzip_bytes','config']}
    result.update(inputs=inputs,inference='Pure cosine on actual WASM; no aliases, query rules or reranking',
                  fixtures={key:score_fixture(v,lookup,f) for key,f in cases.items()})
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(model=name,scores={k:dict(ndcg=s['ndcg@10'],top1=s['top1']) for k,s in result['fixtures'].items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True)
    p.add_argument('--output',type=Path,default=OUT);p.add_argument('--challenge',type=Path);args=p.parse_args()
    for item in args.model:
        name,path=item.split('=',1);evaluate(name,Path(path),args.output,args.challenge)
