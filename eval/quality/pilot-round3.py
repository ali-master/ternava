"""Evaluate frozen fresh/regression retrieval on the actual pilot WASM artifacts."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from bilingual import retrieval_metrics, retrieval_details

ROOT=Path(__file__).resolve().parents[2]



def main(round_number=3):
    out=ROOT/f'dist/pilot-fa-en-v{round_number}'
    destination=out/'retrieval.json'
    if destination.exists():raise FileExistsError('Preserve consumed challenge results')
    models={
        'mini':ROOT/'examples/docs-search/wasm/tern_engine.js',
        'base':ROOT/'examples/emoji-search/wasm/tern_engine.js',
        'round1':ROOT/'dist/pilot-fa-en/d256/tern_engine.js',
        'round2':ROOT/'dist/pilot-fa-en-v2/d256/tern_engine.js',
        'round3':ROOT/'dist/pilot-fa-en-v3/d256/tern_engine.js',
    }
    if round_number==4:models['round4']=out/'d256/tern_engine.js'
    fixtures={f'round{i}':ROOT/f'eval/quality/corpus/bilingual-hard-round{i}.json' for i in range(2,round_number+1)}
    loaded={name:json.loads(path.read_text()) for name,path in fixtures.items()}
    texts=list(dict.fromkeys(r['text'] for f in loaded.values() for r in f['corpus']+f['queries']))
    mapping={t:i for i,t in enumerate(texts)}
    report={'fixtures':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in fixtures.items()},'models':{},'inference':'Pure cosine over actual WASM embeddings; no aliases, reranker, calculator or query rules'}
    for name,glue in models.items():
        raw=subprocess.run(['node',str(ROOT/'eval/quality/collect-wasm.mjs'),str(glue)],input=json.dumps(texts),text=True,capture_output=True,check=True)
        result=json.loads(raw.stdout);vectors=np.asarray(result['embeddings'],dtype=np.float32)
        assert vectors.shape==(len(texts),384) and np.isfinite(vectors).all()
        if name==f'round{round_number}':
            packed=ROOT/f'training/pack/out/pilot-fa-en-v{round_number}/model.bin'
            assert result['model_sha256']==hashlib.sha256(packed.read_bytes()).hexdigest()
            assert result['tokenizer_sha256']=='dddc3fca56fcd694e8b2268d719b7f73827f8fd73d06aedc2e0304cdfc6f08fe'
        measured={k:result[k] for k in ['wasm_sha256','model_sha256','tokenizer_sha256','wasm_bytes','gzip_bytes','config','runtime']}
        measured['fixtures']={}
        for fixture_name,fixture in loaded.items():
            languages={}
            for lang in ['fa','en']:
                corpus=[r for r in fixture['corpus'] if r['lang']==lang]
                queries=[r for r in fixture['queries'] if r['lang']==lang]
                docs={d['id']:i for i,d in enumerate(corpus)}
                scores=vectors[[mapping[q['text']] for q in queries]]@vectors[[mapping[d['text']] for d in corpus]].T
                rel=[{docs[k]:v for k,v in q['relevant'].items()} for q in queries]
                details=retrieval_details(scores,queries,corpus)
                languages[lang]={**retrieval_metrics(scores,rel),**details,'strict_top1':sum(q['top1_correct'] and q['contrast_correct'] for q in details['queries'])}
            measured['fixtures'][fixture_name]=languages
        report['models'][name]=measured
        print(json.dumps({'model':name,'gzip':measured['gzip_bytes'],'scores':{f:{l:v['ndcg@10'] for l,v in ls.items()} for f,ls in measured['fixtures'].items()}}),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,choices=[3,4],default=3,dest='round_number')
    main(parser.parse_args().round_number)
