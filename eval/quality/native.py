"""Frozen native/test/demo evaluation on actual WASM, without aliases."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from bilingual import retrieval_metrics, retrieval_details

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'training/distill/cache/mini-fa-en-round3-native'
OUT=ROOT/'dist/pilot-fa-en-v3-native'


def evaluate(name,glue):
    destination=OUT/f'eval-{name}.json'
    if destination.exists():raise FileExistsError('Preserve consumed test results')
    paths=[CACHE/'qa-test.json',CACHE/'texts.json',ROOT/'eval/quality/corpus/persian-native-demos.json',
           ROOT/'dist/pilot-fa-en/external-sts.json']+[ROOT/f'eval/quality/corpus/bilingual-hard-round{i}.json' for i in [2,3,4]]
    qa,rows,demos,sts,*old=[json.loads(p.read_text()) for p in paths]
    fixtures={'native_qa':qa}
    for task,corpus in demos['corpora'].items():
        fixtures[task]=dict(corpus=corpus,queries=demos['queries'][task])
    for i,fixture in zip([2,3,4],old):
        for lang in ['fa','en']:
            fixtures[f'round{i}_{lang}']={k:[r for r in fixture[k] if r['lang']==lang] for k in ['corpus','queries']}
    # Include English STS strings to prove vector equality, not merely equal correlation.
    english=list(dict.fromkeys(r[k] for r in sts['benchmarks']['en']['samples'] for k in ['sentence1','sentence2']))
    texts=list(dict.fromkeys([r['text'] for f in fixtures.values() for r in f['corpus']+f['queries']]
                +[r[k] for r in rows['qqp_test'] for k in ['q1','q2']]
                +[r[k] for r in rows['nli_test'] for k in ['anchor','positive','negative']]+english))
    mapping={t:i for i,t in enumerate(texts)}
    raw=subprocess.run(['node',str(ROOT/'eval/quality/collect-wasm.mjs'),str(glue)],input=json.dumps(texts),text=True,capture_output=True,check=True)
    result=json.loads(raw.stdout);v=np.asarray(result['embeddings'],dtype=np.float32)
    assert v.shape==(len(texts),384) and np.isfinite(v).all()
    measured={k:result[k] for k in ['model_sha256','tokenizer_sha256','wasm_sha256','wasm_bytes','gzip_bytes','config']}
    measured['inputs']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    measured['inference']='Pure cosine, actual WASM; no aliases, reranking or query rules'
    measured['fixtures']={}
    for key,f in fixtures.items():
        corpus=f['corpus'];queries=f['queries'];lookup={d['id']:i for i,d in enumerate(corpus)}
        scores=v[[mapping[q['text']] for q in queries]]@v[[mapping[d['text']] for d in corpus]].T
        rel=[{lookup[k]:grade for k,grade in q['relevant'].items()} for q in queries]
        metrics=retrieval_metrics(scores,rel)
        order=np.argsort(-scores,axis=1,kind='stable')
        metrics['count']=len(queries);metrics['corpus_count']=len(corpus)
        metrics['top1']=float(np.mean([ranked[0] in r for ranked,r in zip(order,rel)]))
        metrics['queries']=[dict(text=q['text'],top_id=corpus[ranked[0]]['id'],top_text=corpus[ranked[0]]['text'],
                           relevant=q['relevant'],first_relevant_rank=next(i+1 for i,j in enumerate(ranked) if j in r))
                           for q,ranked,r in zip(queries,order,rel)]
        if key.startswith('round'):metrics['contrasts']=retrieval_details(scores,queries,corpus)
        measured['fixtures'][key]=metrics
    qq=rows['qqp_test'];sim=[float(v[mapping[r['q1']]]@v[mapping[r['q2']]]) for r in qq]
    measured['qqp_test']=dict(count=len(qq),spearman=float(spearmanr(sim,[int(r['label']) for r in qq]).statistic))
    nli=rows['nli_test'];gaps=[float(v[mapping[r['anchor']]]@(v[mapping[r['positive']]]-v[mapping[r['negative']]])) for r in nli]
    measured['nli_test']=dict(count=len(nli),correct=sum(g>1e-6 for g in gaps),ties=sum(abs(g)<=1e-6 for g in gaps),gaps=gaps)
    ev=v[[mapping[t] for t in english]]
    ep=OUT/f'english-{name}.npy';np.save(ep,ev)
    if name=='candidate':
        reference=json.loads((OUT/'eval-round3.json').read_text())
        assert measured['inputs']==reference['inputs'], 'Comparison fixtures changed'
        packed=ROOT/'training/pack/out/pilot-fa-en-v3-native/model.bin'
        assert measured['model_sha256']==hashlib.sha256(packed.read_bytes()).hexdigest()
        assert measured['tokenizer_sha256']==reference['tokenizer_sha256']
        baseline=np.load(OUT/'english-round3.npy')
        measured['english_vector_retention']=dict(texts=len(english),exact_equal=bool(np.array_equal(ev,baseline)),max_abs_diff=float(np.max(np.abs(ev-baseline))))
    destination.write_text(json.dumps(measured,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(model=name,scores={k:dict(ndcg=v['ndcg@10'],top1=v['top1']) for k,v in measured['fixtures'].items()},qqp=measured['qqp_test'],nli=measured['nli_test']['correct'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True)
    for item in p.parse_args().model:
        name,glue=item.split('=',1);evaluate(name,Path(glue))
