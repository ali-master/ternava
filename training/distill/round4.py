"""Round-four QAT: paired counterfactual margins, held-out wording, retention gates."""
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from model import StudentEncoder
from loss import distillation_loss, relational_loss
from round2 import encode, take, cosent_loss
from round3 import ranking_loss, validation_fixture, evaluate
from prep.round2 import encode_rows
from prep.round4 import OUT as CACHE, PREVIOUS, ROUND3, CHALLENGE, ROOT
from tokenization import validate_tokenizer, load_tokenizer
import ternary_qat

SOURCE=ROOT/'training/distill/runs/mini-fa-en-round3/best.pt'
TARGETS=['fa_negation','fa_numbers','en_negation','en_numbers']


def contrast_loss(vectors):
    gap=(vectors[:,0]*(vectors[:,1]-vectors[:,2])).sum(-1)
    return F.softplus((.06-gap)*16).mean()/16


def paired_indices(groups, pairs=3):
    # Adjacent examples reverse the same answer pair: sample both or the shortcut returns.
    starts=torch.cat([g[torch.randint(len(g),(pairs,))] for g in groups])
    return torch.stack([starts,starts+1],dim=1).flatten()


def eligible(metrics, baseline):
    return (metrics['sts_en']>=baseline['sts_en']*.995 and metrics['sts_fa']>=baseline['sts_fa']*.99
            and all(metrics['r3_'+l+'_ndcg']>=baseline['r3_'+l+'_ndcg']*.99 for l in ['fa','en'])
            and all(metrics[k]>=baseline[k] for k in TARGETS))


def selection_score(metrics):
    # Worst target group matters independently; a ranking gain cannot hide a numerical loss.
    return min(metrics[k] for k in TARGETS)+.25*sum(metrics[k] for k in TARGETS)+.1*(metrics['fa_ndcg']+metrics['en_ndcg'])


def train(smoke=False):
    random.seed(404);np.random.seed(404);torch.manual_seed(404);torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    manifest=json.loads((CACHE/'manifest.json').read_text());validate_tokenizer(manifest['tokenizer'])
    assert hashlib.sha256(CHALLENGE.read_bytes()).hexdigest()==manifest['challenge_sha256']
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert source['tokenizer']['sha256']==manifest['tokenizer']['sha256']
    old=torch.load(PREVIOUS/'data.pt',weights_only=False);r3=torch.load(ROUND3/'data.pt',weights_only=False);data=torch.load(CACHE/'data.pt',weights_only=False)
    tk=load_tokenizer(manifest['tokenizer']);fixtures={};encoded={}
    for tag,path in [('new',CACHE),('r3',ROUND3)]:
        fixtures[tag]=validation_fixture(json.loads((path/'texts.json').read_text())['val'])
        batch=encode_rows(fixtures[tag]['corpus']+fixtures[tag]['queries'],['text'],tk)
        encoded[tag]={k:v[:,0] for k,v in batch.items()}
    cfg={**source['config'],'run_name':'mini-fa-en-round4','seed':404,'epochs':6,'lr':.000015,'dropout':.03}
    run=ROOT/'training/distill/runs'/(cfg['run_name']+('-smoke' if smoke else ''))
    if run.exists() and not smoke:raise FileExistsError('Preserve completed training runs')
    run.mkdir(parents=True,exist_ok=True);(run/'validation-fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False,indent=2)+'\n')
    args={k:cfg[k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']}
    model=StudentEncoder(**args);model.load_state_dict(source['model_state']);ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model.to(device)
    anchor=StudentEncoder(**args);anchor.load_state_dict(source['model_state']);ternary_qat.swap(anchor);ternary_qat.set_lambda(anchor,1);anchor.to(device).eval().requires_grad_(False)
    def validation():
        m=evaluate(model,old,fixtures['new'],encoded['new'],device)
        previous=evaluate(model,old,fixtures['r3'],encoded['r3'],device)
        m.update({'r3_'+k:v for k,v in previous.items()});m['selection_score']=selection_score(m)
        return m
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01);steps=500
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100,num_training_steps=steps*cfg['epochs'])
    pairgroups=[torch.where(data['train']['group'][::2]==g)[0]*2 for g in [0,1,3,4]]
    groups=[torch.where(r3['train']['group']==g)[0] for g in range(7)]
    sts_groups=[torch.where(old['sts_train']['lang']==g)[0] for g in [0,1]]
    baseline=validation() if not smoke else {**{k:0 for k in TARGETS},'sts_en':0,'sts_fa':0,'fa_ndcg':0,'en_ndcg':0,'r3_fa_ndcg':0,'r3_en_ndcg':0,'selection_score':0}
    (run/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');print('BASELINE',baseline,flush=True)
    recipe=dict(steps_per_epoch=steps,replay_batch=48,counterfactual_batch=24,round3_ranking_batch=28,sts_batch=32,loss='0.3 new listwise + 0.4 round3 listwise + 2 explicit contrast margin + 0.8 round3 replay cosine + 0.3 round3 triplet cosine + 0.2 teacher cosine + 0.1 relational + 0.03 CoSENT',selection='Worst-group and mean target accuracy with NDCG; per-language old ranking >=99%, English STS >=99.5%, FA STS >=99%, all four target contrast accuracies >= baseline; int4 validation only')
    best=-float('inf');best_epoch=None;diagnostic=-float('inf')
    for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
        start=time.monotonic();model.train();losses=[]
        for step in range(1 if smoke else steps):
            ranked=take(data['train'],paired_indices(pairgroups));previous=take(r3['train'],torch.cat([g[torch.randint(len(g),(4,))] for g in groups]))
            replay=take(old['replay'],torch.randint(len(old['replay']['input_ids']),(48,)))
            scored=take(old['sts_train'],torch.cat([g[torch.randint(len(g),(16,))] for g in sts_groups]))
            v=encode(model,ranked,device);p=encode(model,previous,device);r=encode(model,replay,device);s=encode(model,scored,device)
            with torch.no_grad():target=encode(anchor,replay,device);previous_target=encode(anchor,previous,device)
            rank=ranking_loss(v,ranked['topic'].to(device),ranked['doc_ids'].to(device),ranked['group'].to(device))
            oldrank=ranking_loss(p,previous['topic'].to(device),previous['doc_ids'].to(device),previous['group'].to(device))
            contrast=contrast_loss(v);retain=distillation_loss(r,target);teacher=replay['teacher_emb'].to(device)
            tags=scored['lang'].to(device);scores=scored['score'].to(device)
            sts=sum(cosent_loss(s[tags==tag],scores[tags==tag]) for tag in [0,1])/2
            loss=.3*rank+.4*oldrank+2*contrast+.8*retain+.3*distillation_loss(p.flatten(0,1),previous_target.flatten(0,1))+.2*distillation_loss(r,teacher)+.1*relational_loss(r,target)+.03*sts
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();scheduler.step();losses.append(float(loss.detach()))
            if (step+1)%100==0 or smoke:print(f'epoch={epoch} step={step+1} loss={float(loss):.4f} rank={float(rank):.4f} oldrank={float(oldrank):.4f} contrast={float(contrast):.4f} retain={float(retain):.4f}',flush=True)
        metrics=validation() if not smoke else baseline.copy()
        metrics.update(epoch=epoch,eligible=eligible(metrics,baseline),seconds=time.monotonic()-start,loss=float(np.mean(losses)))
        print('EPOCH',metrics,flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
        checkpoint=dict(epoch=epoch,model_state={k:v.cpu() for k,v in model.state_dict().items()},config=cfg,tokenizer=source['tokenizer'],data_manifest=manifest,source_checkpoint_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),validation=metrics,recipe=recipe)
        if metrics['selection_score']>diagnostic:diagnostic=metrics['selection_score'];torch.save(checkpoint,run/'diagnostic-best.pt')
        if metrics['eligible'] and metrics['selection_score']>best:best=metrics['selection_score'];best_epoch=epoch;torch.save(checkpoint,run/'best.pt')
        if epoch==cfg['epochs'] or smoke:torch.save(checkpoint,run/f'checkpoint_ep{epoch}.pt')
    if not smoke:
        print(f'VERDICT: {"WIN" if best>baseline["selection_score"] else "REGRESSION"} | score={best:.5f} | baseline={baseline["selection_score"]:.5f} | epoch={best_epoch} | diagnostic={diagnostic:.5f}',flush=True)
        if best_epoch is None:raise RuntimeError('No eligible checkpoint; diagnostic-best is not accepted')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');train(parser.parse_args().smoke)
