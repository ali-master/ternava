"""Paired margin QAT with separate domain and English geometry teachers."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import ternary_qat
from domain import evaluate, filter_triplets, listwise_loss, replay_groups, validation_fixtures
from model import StudentEncoder
from native import embed_all, int4_ste
from prep.domain import CACHE as DOMAIN
from prep.native import ROOT, digest
from prep.reasoning import CACHE, CHALLENGE, SOURCE
from prep.round2 import encode_rows
from round2 import encode, take
from round3 import ranking_loss
from tokenization import load_tokenizer

EN_SOURCE=ROOT/'training/distill/runs/mini-fa-en-round3/best.pt'


def paired_loss(v):
    assert len(v)%2==0
    gaps=(v[:,0]*(v[:,1]-v[:,2])).sum(-1).reshape(-1,2)
    # A fixed preferred answer has opposite margins; penalize the worse direction too.
    individual=F.softplus((.035-gaps)*32).mean()
    worst=F.softplus((.035-gaps.min(dim=1).values)*32).mean()
    direction=1-F.cosine_similarity(v[::2,0]-v[1::2,0],v[::2,1]-v[::2,2],dim=-1,eps=1e-5)
    return individual+worst+.25*direction.mean()


def geometry_loss(v,target):
    assert v.ndim==target.ndim==2 and v.shape==target.shape
    point=(1-(v*target).sum(-1)).mean()
    scores=v@v.T;gold=target@target.T
    return 30*point+10*(scores-gold).square().mean()+F.kl_div(F.log_softmax(scores*16,dim=-1),F.softmax(gold*16,dim=-1),reduction='batchmean')


def load_model(ckpt,device,train=False):
    args={k:ckpt['config'][k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','position_embeddings']}
    model=StudentEncoder(**args,dropout=0);model.load_state_dict(ckpt['model_state']);ternary_qat.swap(model);ternary_qat.set_lambda(model,1)
    if train:model.embedding.register_forward_hook(int4_ste)
    else:ternary_qat.int4_quantize_embedding_(model);model.requires_grad_(False)
    return model.to(device)


def fixtures():
    result=validation_fixtures()
    prior=json.loads((ROOT/'eval/quality/corpus/persian-domain-hard.json').read_text())
    for k,f in prior['suites'].items():result['domain_'+k]=f
    return result


@torch.no_grad()
def assess(model,cases,tokens,sts,logic,device):
    result=evaluate(model,cases,tokens,sts,device)
    original=model.embedding.weight.detach().clone();ternary_qat.int4_quantize_embedding_(model)
    try:
        v=embed_all(model,logic,device);gap=(v[:,0]*(v[:,1]-v[:,2])).sum(-1)
        for g,name in enumerate(['fa_negation','fa_numbers','en_negation','en_numbers']):
            mask=logic['group']==g;wins=gap[mask]>1e-6
            result[name+'_contrast']=float(wins.float().mean());result[name+'_paired']=float(wins.reshape(-1,2).all(-1).float().mean())
        result['selection_score']=sum(result[k+'_paired'] for k in ['fa_negation','fa_numbers'])/2+.25*(result['domain_docs_ndcg']+result['domain_emoji_ndcg'])
        return result
    finally:model.embedding.weight.copy_(original)


def eligible(m,b,en):
    return (m['sts_en']>=en['sts_en']-.002 and m['sts_fa']>=b['sts_fa']-.005
        and all(m[f'r{i}_en_ndcg']>=en[f'r{i}_en_ndcg']-1e-6 for i in [2,3,4])
        and all(m[f'r{i}_fa_ndcg']>=b[f'r{i}_fa_ndcg']-.01 for i in [2,3,4])
        and all(m['domain_'+k+'_top1']>=b['domain_'+k+'_top1'] for k in ['docs','emoji','unseen_corpus'])
        and all(m[k]>=b[k] for k in ['negation','numbers','fa_negation_paired','fa_numbers_paired']))


def train(smoke=False):
    torch.manual_seed(719);np.random.seed(719);torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    run=ROOT/'training/distill/runs'/('mini-fa-en-reasoning-smoke' if smoke else 'mini-fa-en-reasoning')
    if run.exists():raise FileExistsError('Preserve every experiment')
    manifest=json.loads((CACHE/'manifest.json').read_text())
    assert digest(CHALLENGE)==manifest['challenge_sha256'] and digest(SOURCE)==manifest['source_sha256']
    for file,sha in manifest['files'].items():assert digest(CACHE/file)==sha
    source=torch.load(SOURCE,map_location='cpu',weights_only=False);english=torch.load(EN_SOURCE,map_location='cpu',weights_only=False)
    assert digest(source['tokenizer']['path'])==digest(english['tokenizer']['path'])==manifest['tokenizer_sha256']
    tk=load_tokenizer(source['tokenizer']);model=load_model(source,device,True)
    teachers=[load_model(c,device).eval() for c in [english,source]]
    logic=torch.load(CACHE/'data.pt',weights_only=False);domain=torch.load(DOMAIN/'data.pt',weights_only=False)
    old=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False)
    replay_texts=json.loads((ROOT/'training/distill/cache/mini-fa-en-round2/texts.json').read_text())['replay_texts']
    opus=json.loads((ROOT/'training/distill/cache/mini-fa-en-32k/pairs.json').read_text())['train'];replay_ix=replay_groups(replay_texts,opus)
    r3dir=ROOT/'training/distill/cache/mini-fa-en-round3';r3=filter_triplets(torch.load(r3dir/'data.pt',weights_only=False)['train'],json.loads((r3dir/'texts.json').read_text())['train'],set())
    en_triplets=torch.where(r3['group']>=3)[0]
    groups=[torch.where(logic['train']['group'][::2]==g)[0]*2 for g in range(4)]
    cases=fixtures();tokens={k:encode_rows(f['corpus']+f['queries'],['text'],tk) for k,f in cases.items()}
    run.mkdir(parents=True)
    recipe=dict(seed=719,epochs=6,steps=300,embedding_lr=.00012,body_lr=.000012,paired_batch=32,
                trainable='All existing weights including numeric tokens; no auxiliary inference parameters',
                loss='1 domain listwise + 1 paired worst-direction margin + .5 old English rank + bilingual dual-teacher geometry',
                source_sha256=digest(SOURCE),english_teacher_sha256=digest(EN_SOURCE),challenge_sha256=digest(CHALLENGE),
                selection='Validation paired FA logic plus .25 domain NDCG. All historical FA gates, English ranking >= round3 on every set. No fresh-test selection.')
    (run/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    baseline=assess(model,cases,tokens,old['sts_val'],logic['val'],device)
    enbase=assess(teachers[0],cases,tokens,old['sts_val'],logic['val'],device)
    (run/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');(run/'english-baseline.json').write_text(json.dumps(enbase,indent=2)+'\n')
    print('BASELINE',baseline,flush=True)
    optimizer=torch.optim.AdamW([{'params':[model.embedding.weight],'lr':recipe['embedding_lr']},
        {'params':[p for n,p in model.named_parameters() if n!='embedding.weight'],'lr':recipe['body_lr']}],weight_decay=0)
    best=diagnostic=-float('inf')
    for epoch in range(1,(1 if smoke else recipe['epochs'])+1):
        start=time.monotonic();model.train();losses=[]
        for step in range(1 if smoke else recipe['steps']):
            starts=torch.cat([g[torch.randint(len(g),(4,))] for g in groups]);ix=torch.stack([starts,starts+1],1).flatten()
            lv=encode(model,take(logic['train'],ix),device);reason=paired_loss(lv)
            task=['docs','emoji'][step%2];batch=take(domain[task+'_train'],torch.randint(len(domain[task+'_train']['input_ids']),(20,)))
            q=encode(model,batch,device)[:,0];d=encode(model,domain[task+'_corpus'],device)[:,0];rank=listwise_loss(q,d,batch['relevant'].to(device))
            with torch.no_grad():tq=encode(teachers[1],batch,device)[:,0];td=encode(teachers[1],domain[task+'_corpus'],device)[:,0]
            retention=geometry_loss(torch.cat([q,d]),torch.cat([tq,td]))
            for lang,teacher in enumerate(teachers):
                ids=replay_ix[lang];b=take(old['replay'],ids[torch.randint(len(ids),(32,))]);v=encode(model,b,device)
                with torch.no_grad():tv=encode(teacher,b,device)
                retention=retention+geometry_loss(v,tv)*(2 if lang==0 else .5)
            b=take(r3,en_triplets[torch.randint(len(en_triplets),(24,))]);v=encode(model,b,device)
            oldrank=ranking_loss(v,b['topic'].to(device),b['doc_ids'].to(device),b['group'].to(device))
            with torch.no_grad():tv=encode(teachers[0],b,device)
            retention=retention+geometry_loss(v.flatten(0,1),tv.flatten(0,1))
            loss=reason+rank+.5*oldrank+retention
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            progress=(epoch-1)*recipe['steps']+step;factor=min(1,(progress+1)/50)*(1-progress/(recipe['epochs']*recipe['steps']))
            for group,lr in zip(optimizer.param_groups,[recipe['embedding_lr'],recipe['body_lr']]):group['lr']=lr*factor
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%50==0 or smoke:print(f'epoch={epoch} step={step+1} loss={losses[-1]:.4f} reasoning={float(reason.detach()):.4f} domain={float(rank.detach()):.4f} retention={float(retention.detach()):.4f}',flush=True)
        m=assess(model,cases,tokens,old['sts_val'],logic['val'],device) if not smoke else baseline.copy()
        m.update(epoch=epoch,eligible=eligible(m,baseline,enbase),seconds=time.monotonic()-start,loss=float(np.mean(losses)))
        print('EPOCH',m,flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
        ckpt=dict(epoch=epoch,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config={**source['config'],'run_name':'mini-fa-en-reasoning'},tokenizer=source['tokenizer'],validation=m,recipe=recipe,data_manifest=manifest)
        if m['selection_score']>diagnostic:diagnostic=m['selection_score'];torch.save(ckpt,run/'diagnostic-best.pt')
        if m['eligible'] and m['selection_score']>best:best=m['selection_score'];torch.save(ckpt,run/'best.pt')
        if epoch==recipe['epochs'] or smoke:torch.save(ckpt,run/f'checkpoint_ep{epoch}.pt')
    print('VERDICT',dict(eligible=best!=-float('inf'),diagnostic_score=diagnostic,baseline=baseline['selection_score']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');train(p.parse_args().smoke)
