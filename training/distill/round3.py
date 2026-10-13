"""Fixed-size round-three QAT: listwise retrieval with round-two retention."""
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

from model import StudentEncoder
from loss import distillation_loss, relational_loss
from round2 import encode, take, cosent_loss
from prep.round3 import OUT as CACHE, PREVIOUS, CHALLENGE, ROOT, normalize
from tokenization import validate_tokenizer
import ternary_qat

SOURCE=ROOT/'training/distill/runs/mini-fa-en-round2-curriculum/best.pt'


def ranking_loss(vectors, topics, doc_ids, groups):
    """Own contradiction + other-topic documents; duplicate positives never repel."""
    q=vectors[:,0]; docs=vectors[:,1:].reshape(-1,vectors.shape[-1])
    scores=q@docs.T*12
    lang=(groups>=3).long()
    allowed=(topics[:,None]!=topics.repeat_interleave(2)[None,:]) & (lang[:,None]==lang.repeat_interleave(2)[None,:])
    same_positive=doc_ids[:,0,None]==doc_ids.flatten()[None,:]
    allowed &= ~same_positive
    indices=torch.arange(len(q),device=q.device)
    allowed[indices,indices*2]=True
    allowed[indices,indices*2+1]=True
    return F.cross_entropy(scores.masked_fill(~allowed,float('-inf')),indices*2)


def validation_fixture(rows):
    rng=random.Random(304); selected=[]
    for lang in ['fa','en']:
        for cat in ['negation','numbers','role-order','nli']:
            subset=[r for r in rows if r['lang']==lang and r['category']==cat]
            rng.shuffle(subset);selected+=subset[:36 if cat!='nli' else 100]
    corpus=[];mapping={};queries=[]
    for row in selected:
        ids=[]
        for key in ['positive','negative']:
            text=row[key];canonical=(row['lang'],normalize(text))
            if canonical not in mapping:
                mapping[canonical]=len(corpus);corpus.append({'text':text,'lang':row['lang']})
            ids.append(mapping[canonical])
        queries.append(dict(text=row['anchor'],lang=row['lang'],category=row['category'],positive=ids[0],negative=ids[1]))
    return dict(corpus=corpus,queries=queries)


@torch.no_grad()
def evaluate(model, old, fixture, encoded, device):
    model.eval()
    # Selection must reflect the actual int4 artifact, then restore trainable weights.
    original=model.embedding.weight.detach().clone()
    ternary_qat.int4_quantize_embedding_(model)
    try:
        v=torch.cat([encode(model,take(encoded,slice(i,i+64)),device).cpu() for i in range(0,len(encoded['input_ids']),64)])
        nc=len(fixture['corpus']);metrics={}
        for lang in ['fa','en']:
            candidates=[i for i,d in enumerate(fixture['corpus']) if d['lang']==lang]
            ndcg=[];top=[];categories={}
            for i,q in enumerate(fixture['queries']):
                if q['lang']!=lang:continue
                scores=(v[nc+i]@v[candidates].T).numpy()
                rank=list(np.argsort(-scores,kind='stable')).index(candidates.index(q['positive']))+1
                ndcg.append(1/np.log2(rank+1) if rank<=10 else 0);top.append(rank==1)
                gap=float(v[nc+i]@(v[q['positive']]-v[q['negative']]))
                categories.setdefault(q['category'],[]).append(gap>1e-6)
            metrics[lang+'_ndcg']=float(np.mean(ndcg));metrics[lang+'_top1']=float(np.mean(top))
            for cat,values in categories.items():metrics[lang+'_'+cat]=float(np.mean(values))
        sim=[]
        for i in range(0,len(old['sts_val']['input_ids']),64):
            e=encode(model,take(old['sts_val'],slice(i,i+64)),device)
            sim.append((e[:,0]*e[:,1]).sum(-1).cpu())
        sim=torch.cat(sim)
        for lang,tag in [('en',0),('fa',1)]:
            keep=old['sts_val']['lang']==tag
            metrics['sts_'+lang]=float(spearmanr(sim[keep],old['sts_val']['score'][keep]).statistic)
        metrics['selection_score']=(metrics['fa_ndcg']+metrics['en_ndcg'])/2 + .05*sum(metrics[l+'_'+c] for l in ['fa','en'] for c in ['negation','numbers'])
        assert all(np.isfinite(v) for v in metrics.values())
        return metrics
    finally:
        model.embedding.weight.copy_(original)


def train(smoke=False):
    random.seed(303);np.random.seed(303);torch.manual_seed(303);torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    manifest=json.loads((CACHE/'manifest.json').read_text());validate_tokenizer(manifest['tokenizer'])
    assert hashlib.sha256(CHALLENGE.read_bytes()).hexdigest()==manifest['challenge_sha256']
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert source['tokenizer']['sha256']==manifest['tokenizer']['sha256']
    old=torch.load(PREVIOUS/'data.pt',weights_only=False)
    data=torch.load(CACHE/'data.pt',weights_only=False)
    rows=json.loads((CACHE/'texts.json').read_text())
    from prep.round2 import encode_rows
    from tokenization import load_tokenizer
    fixture=validation_fixture(rows['val'])
    encoded=encode_rows(fixture['corpus']+fixture['queries'],['text'],load_tokenizer(manifest['tokenizer']))
    encoded={k:v[:,0] for k,v in encoded.items()}
    cfg={**source['config'],'run_name':'mini-fa-en-round3','seed':303,'epochs':6,'lr':0.00003,'dropout':.05}
    run=ROOT/'training/distill/runs'/(cfg['run_name']+('-smoke' if smoke else ''))
    if run.exists() and not smoke:raise FileExistsError('Preserve completed training runs')
    run.mkdir(parents=True,exist_ok=True)
    (run/'validation-fixture.json').write_text(json.dumps(fixture,ensure_ascii=False,indent=2)+'\n')
    args={k:cfg[k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']}
    model=StudentEncoder(**args);model.load_state_dict(source['model_state'])
    ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model.to(device)
    anchor=StudentEncoder(**args);anchor.load_state_dict(source['model_state'])
    ternary_qat.swap(anchor);ternary_qat.set_lambda(anchor,1);anchor.to(device).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    steps=625
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=125,num_training_steps=steps*cfg['epochs'])
    groups=[torch.where(data['train']['group']==g)[0] for g in range(7)]
    sts_groups=[torch.where(old['sts_train']['lang']==g)[0] for g in [0,1]]
    baseline=evaluate(model,old,fixture,encoded,device) if not smoke else {'selection_score':0,'sts_en':0,'sts_fa':0}
    (run/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');print('BASELINE',baseline,flush=True)
    recipe=dict(steps_per_epoch=steps,replay_batch=48,ranking_batch=32,sts_batch=32,ranking_weight=.4,round2_retention_weight=.8,teacher_weight=.2,relational_weight=.1,sts_weight=.03,selection='Max full-corpus validation NDCG plus .05 sum of four target contrast accuracies, subject to en STS >=99% and fa STS >=98% of round2 validation; int4 selection; no test tuning')
    best=-float('inf');best_epoch=None
    for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
        start=time.monotonic();model.train();losses=[]
        for step in range(1 if smoke else steps):
            ids=torch.cat([g[torch.randint(len(g),(4 if i<6 else 8,))] for i,g in enumerate(groups)])
            ranked=take(data['train'],ids)
            replay=take(old['replay'],torch.randint(len(old['replay']['input_ids']),(48,)))
            scored=take(old['sts_train'],torch.cat([g[torch.randint(len(g),(16,))] for g in sts_groups]))
            v=encode(model,ranked,device);r=encode(model,replay,device);s=encode(model,scored,device)
            with torch.no_grad():target=encode(anchor,replay,device)
            rank=ranking_loss(v,ranked['topic'].to(device),ranked['doc_ids'].to(device),ranked['group'].to(device))
            retain=distillation_loss(r,target)
            teacher=replay['teacher_emb'].to(device)
            distill=.2*distillation_loss(r,teacher)+.1*relational_loss(r,target)
            tags=scored['lang'].to(device);scores=scored['score'].to(device)
            sts=sum(cosent_loss(s[tags==tag],scores[tags==tag]) for tag in [0,1])/2
            loss=.4*rank+.8*retain+distill+.03*sts
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
            losses.append(float(loss.detach()))
            if (step+1)%100==0 or smoke:print(f'epoch={epoch} step={step+1} loss={float(loss):.4f} rank={float(rank):.4f} retain={float(retain):.4f} sts={float(sts):.4f}',flush=True)
        metrics=evaluate(model,old,fixture,encoded,device) if not smoke else baseline.copy()
        eligible=metrics['sts_en']>=baseline['sts_en']*.99 and metrics['sts_fa']>=baseline['sts_fa']*.98
        metrics.update(epoch=epoch,eligible=eligible,seconds=time.monotonic()-start,loss=float(np.mean(losses)))
        print('EPOCH',metrics,flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        checkpoint=dict(epoch=epoch,model_state={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,tokenizer=source['tokenizer'],data_manifest=manifest,source_checkpoint_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),validation=metrics,recipe=recipe)
        if eligible and metrics['selection_score']>best:
            best=metrics['selection_score'];best_epoch=epoch;torch.save(checkpoint,run/'best.pt')
        if epoch==cfg['epochs'] or smoke:torch.save(checkpoint,run/f'checkpoint_ep{epoch}.pt')
    if not smoke:
        if best_epoch is None:raise RuntimeError('No checkpoint preserved validation STS; final checkpoint remains for diagnosis only')
        print(f'VERDICT: {"WIN" if best>baseline["selection_score"] else "REGRESSION"} | score={best:.5f} | baseline={baseline["selection_score"]:.5f} | delta={best-baseline["selection_score"]:.5f} | epoch={best_epoch}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');train(parser.parse_args().smoke)
