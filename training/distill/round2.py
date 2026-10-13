"""Warm-started QAT with replay, trusted hard negatives and labeled STS ranking."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import get_linear_schedule_with_warmup

from model import StudentEncoder
from loss import distillation_loss, relational_loss
import ternary_qat
from tokenization import validate_tokenizer

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'training/distill/cache/mini-fa-en-round2'
SOURCE=ROOT/'training/distill/runs/mini-fa-en-32k-pilot-23da804/checkpoint_ep10.pt'

def triplet_loss(vectors, margin=0.1):
    a,p,n=vectors.unbind(dim=1)
    gap=(a*p).sum(-1)-(a*n).sum(-1)
    return F.softplus(10*(margin-gap)).mean()/10

def cosent_loss(vectors,scores):
    sim=(vectors[:,0]*vectors[:,1]).sum(-1)*20
    differences=sim[:,None]-sim[None,:]
    ordered=scores[:,None]<scores[None,:]
    return torch.logsumexp(torch.cat([differences[ordered],sim.new_zeros(1)]),dim=0)

def encode(model, batch, device):
    shape=batch['input_ids'].shape
    ids=batch['input_ids'].reshape(-1,shape[-1]).to(device)
    mask=batch['attention_mask'].reshape_as(ids).to(device)
    result=model(ids,mask)
    return result.reshape(*shape[:-1],-1)

def take(data,index):return {k:v[index] for k,v in data.items()}

@torch.no_grad()
def evaluate(model,data,device):
    model.eval();metrics={}
    triples=[]
    for i in range(0,len(data['triplet_val']['input_ids']),64):
        v=encode(model,take(data['triplet_val'],slice(i,i+64)),device)
        triples.append(((v[:,0]*v[:,1]).sum(-1)-(v[:,0]*v[:,2]).sum(-1)).cpu())
    gaps=torch.cat(triples);metrics['triplet_accuracy']=float((gaps>1e-6).float().mean())
    values=[]
    for i in range(0,len(data['sts_val']['input_ids']),64):
        v=encode(model,take(data['sts_val'],slice(i,i+64)),device)
        values.append((v[:,0]*v[:,1]).sum(-1).cpu())
    values=torch.cat(values)
    for lang,tag in [('en',0),('fa',1)]:
        keep=data['sts_val']['lang']==tag
        metrics['sts_'+lang]=float(spearmanr(values[keep],data['sts_val']['score'][keep]).statistic)
    assert all(np.isfinite(v) for v in metrics.values())
    contrast_score=metrics['triplet_accuracy']
    if 'group' in data['triplet_val']:
        categories=[]
        for group in range(6):
            subset=gaps[data['triplet_val']['group']==group]
            accuracy=float((subset>1e-6).float().mean())
            metrics[f'contrast_group_{group}']=accuracy;categories.append(accuracy)
        contrast_score=sum(categories)/len(categories)
        metrics['contrast_macro']=contrast_score
    metrics['selection_score']=metrics['sts_en']+metrics['sts_fa']+0.5*contrast_score
    return metrics

def train(smoke=False, curriculum=False, stop_after_epochs=None):
    torch.manual_seed(75);np.random.seed(75);random.seed(75);torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    manifest=json.loads((CACHE/'manifest.json').read_text());validate_tokenizer(manifest['tokenizer'])
    data=torch.load(CACHE/'data.pt',weights_only=False)
    source_path=ROOT/'training/distill/runs/mini-fa-en-round2/best.pt' if curriculum else SOURCE
    source=torch.load(source_path,weights_only=False,map_location='cpu')
    assert source['tokenizer']['sha256']==manifest['tokenizer']['sha256']
    cfg={**source['config'],'position_embeddings':True,'dropout':0.05,'epochs':8,'lr':0.00005,'qat_warmup_epochs':0,'run_name':'mini-fa-en-round2','seed':75}
    if curriculum:
        cfg.update(epochs=4,run_name='mini-fa-en-round2-curriculum')
        rows=json.loads((CACHE/'texts.json').read_text())['triplets']
        for split in ['train','val']:
            data['triplet_'+split]['group']=torch.tensor([(0 if r['lang']=='fa' else 3)+['negation','numbers','role-order'].index(r['category']) if r['category']!='nli' else 6 for r in rows[split]])
    if stop_after_epochs is not None and not 1 <= stop_after_epochs <= cfg['epochs']:
        raise ValueError('stop_after_epochs must be within the configured training horizon')
    epochs_to_run=stop_after_epochs or cfg['epochs']
    run=ROOT/'training/distill/runs'/(cfg['run_name']+('-smoke' if smoke else ''))
    if not smoke and run.exists():raise FileExistsError('Do not overwrite an existing round-two run')
    run.mkdir(parents=True,exist_ok=True)
    model=StudentEncoder(**{k:cfg[k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']})
    mismatch=model.load_state_dict(source['model_state'],strict=False)
    assert mismatch.missing_keys==([] if source['config'].get('position_embeddings') else ['position_embedding.weight']) and not mismatch.unexpected_keys
    ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model=model.to(device)
    parameters=[{'params':[p for name,p in model.named_parameters() if not name.startswith('position_embedding.') ]}, {'params':model.position_embedding.parameters(),'lr':cfg['lr']*10}] if curriculum else model.parameters()
    optimizer=torch.optim.AdamW(parameters,lr=cfg['lr'],weight_decay=0.01)
    steps=(len(data['replay']['input_ids'])+63)//64
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(steps*cfg['epochs']*.05),num_training_steps=steps*cfg['epochs'])
    if smoke:
        data={k:{f:v[torch.cat([torch.where(group['group']==g)[0][:8] for g in range(7)])] if 'group' in group else v[:128] for f,v in group.items()} for k,group in data.items()};steps=1
    baseline=evaluate(model,data,device) if not smoke else {}
    (run/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');print('BASELINE',baseline,flush=True)
    sts_indices=[torch.where(data['sts_train']['lang']==tag)[0] for tag in [0,1]]
    if smoke:sts_indices=[torch.arange(len(data['sts_train']['lang']))]*2
    triplet_indices=[torch.where(data['triplet_train']['group']==g)[0] for g in range(7)] if curriculum else []
    best=-float('inf');global_step=0
    for epoch in range(1,(1 if smoke else epochs_to_run)+1):
        start=time.monotonic();model.train();losses=[]
        order=torch.randperm(len(data['replay']['input_ids']))
        for step in range(steps):
            replay=take(data['replay'],order[step*64:(step+1)*64])
            triple_ids=torch.cat([indices[torch.randint(len(indices),(4 if g<6 else 8,))] for g,indices in enumerate(triplet_indices)]) if curriculum else torch.randint(len(data['triplet_train']['input_ids']),(24,))
            triple=take(data['triplet_train'],triple_ids)
            scored=take(data['sts_train'],torch.cat([indices[torch.randint(len(indices),(16,))] for indices in sts_indices]))
            r=encode(model,replay,device);t=encode(model,triple,device);s=encode(model,scored,device)
            target=replay['teacher_emb'].to(device)
            distill=distillation_loss(r,target)+0.1*relational_loss(r,target)
            contrast=triplet_loss(t)
            # Rank pairs within each language, avoiding dataset-specific score scales.
            sts=sum(cosent_loss(s[scored['lang'].to(device)==tag],scored['score'].to(device)[scored['lang'].to(device)==tag]) for tag in [0,1])/2
            loss=distill+(1.5 if curriculum else 0.6)*contrast+0.03*sts
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step()
            global_step+=1;losses.append(float(loss.detach()))
            if global_step%100==0 or smoke:print(f'epoch={epoch} step={global_step} loss={float(loss):.4f} distill={float(distill):.4f} triplet={float(contrast):.4f} sts={float(sts):.4f}',flush=True)
        metrics=evaluate(model,data,device) if not smoke else {'selection_score':0}
        metrics.update(epoch=epoch,seconds=time.monotonic()-start,loss=float(np.mean(losses)))
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        print('EPOCH',metrics,flush=True)
        checkpoint={'epoch':epoch,'model_state':{k:v.cpu() for k,v in model.state_dict().items()},'config':cfg,'tokenizer':source['tokenizer'],'data_manifest':manifest,'source_checkpoint_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'validation':metrics,'recipe':{'replay_batch':64,'triplet_batch':32 if curriculum else 24,'curriculum':curriculum,'position_lr_multiplier':10 if curriculum else 1,'sts_batch':32,'triplet_weight':1.5 if curriculum else .6,'sts_weight':.03,'triplet_margin':.1,'selection':'validation en+fa Spearman + .5 balanced target-category accuracy; no test selection' if curriculum else 'validation en+fa Spearman + .5 triplet accuracy; no test selection'}}
        if metrics['selection_score']>best:
            best=metrics['selection_score'];torch.save(checkpoint,run/'best.pt')
        if epoch==epochs_to_run or smoke:torch.save(checkpoint,run/f'checkpoint_ep{epoch}.pt')
    if not smoke:print(f'VERDICT: {"WIN" if best>baseline["selection_score"] else "REGRESSION"} | score={best:.5f} | baseline={baseline["selection_score"]:.5f} | delta={best-baseline["selection_score"]:.5f}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--curriculum',action='store_true');p.add_argument('--stop-after-epochs',type=int);args=p.parse_args();train(args.smoke,args.curriculum,args.stop_after_epochs)
