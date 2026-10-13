"""Authored domain supervision; fresh queries remain sealed until final WASM evaluation."""
import json
from pathlib import Path

import torch

from prep.native import ROOT, SOURCE, digest
from prep.round2 import encode_rows, normalize
from tokenization import load_tokenizer

CACHE=ROOT/'training/distill/cache/mini-fa-en-domain'
CHALLENGE=ROOT/'eval/quality/corpus/persian-domain-hard.json'
QUERIES=Path(__file__).with_name('domain_queries.tsv')
DOCS_CORPUS='eval/quality/corpus/react-docs.fa.json'


def domain_rows():
    # Preserve the original supervision when the public demo changes topic.
    docs=json.loads((ROOT/DOCS_CORPUS).read_text())
    emoji=json.loads((ROOT/'examples/emoji-search/corpus.fa.json').read_text())
    corpora={'docs':[dict(id=d['id'],text=d['text']) for d in docs],
             'emoji':[dict(id=d['emoji'],text=d['name']+'، '+d['text']) for d in emoji]}
    rows={'train':[],'val':[]};seen=set()
    for line in QUERIES.read_text().splitlines():
        key,*queries=line.split('|');task,identity=key.split(':',1)
        assert len(queries)==4 and key not in seen,key
        seen.add(key)
        identity='fa-react-'+identity if task=='docs' else identity
        targets=[identity]
        if identity in ['🥰','❤️','😍']:targets=['🥰','❤️','😍']
        if identity in ['🥳','🎉']:targets=['🥳','🎉']
        for i,q in enumerate(queries):
            rows['val' if i==3 else 'train'].append(dict(text=q,task=task,relevant=dict.fromkeys(targets,1)))
    assert len(seen)==sum(map(len,corpora.values()))
    return corpora,rows


def prepare():
    if CACHE.exists():raise FileExistsError('Preserve domain data')
    source=torch.load(SOURCE,map_location='cpu',weights_only=False);tk=load_tokenizer(source['tokenizer'])
    corpora,rows=domain_rows()
    challenge=json.loads(CHALLENGE.read_text())
    held={normalize(q['text']) for f in challenge['suites'].values() for q in f['queries']}
    train={normalize(q['text']) for q in rows['train']};val={normalize(q['text']) for q in rows['val']}
    assert not train&val and not held&(train|val)
    # Corpus texts are explicitly in-domain supervision, not claimed unseen documents.
    data={};fixtures={}
    for task,corpus in corpora.items():
        lookup={d['id']:i for i,d in enumerate(corpus)}
        data[task+'_corpus']=encode_rows(corpus,['text'],tk)
        for split in ['train','val']:
            subset=[q for q in rows[split] if q['task']==task]
            batch=encode_rows(subset,['text'],tk)
            mask=torch.zeros(len(subset),len(corpus),dtype=torch.bool)
            for i,q in enumerate(subset):
                for k in q['relevant']:mask[i,lookup[k]]=True
            batch['relevant']=mask;data[task+'_'+split]=batch
        fixtures[task]=dict(corpus=corpus,queries=[q for q in rows['val'] if q['task']==task])
    CACHE.mkdir(parents=True)
    torch.save(data,CACHE/'data.pt')
    (CACHE/'texts.json').write_text(json.dumps(dict(corpora=corpora,**rows),ensure_ascii=False)+'\n')
    (CACHE/'validation.json').write_text(json.dumps(fixtures,ensure_ascii=False)+'\n')
    manifest=dict(tokenizer=source['tokenizer'],source_sha256=digest(SOURCE),challenge_sha256=digest(CHALLENGE),
        authored_queries_sha256=digest(QUERIES),counts={k:len(v) for k,v in rows.items()},
        corpus_sha256={p:digest(ROOT/p) for p in [DOCS_CORPUS,'examples/emoji-search/corpus.fa.json']},
        supervision='Three authored queries per known corpus item; fourth query only for validation. No alias fields used. Fresh queries and unseen-corpus texts excluded from training.',
        files={p.name:digest(p) for p in CACHE.iterdir() if p.is_file()})
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['counts'])


if __name__=='__main__':prepare()
