"""English-invariant QAT: multi-answer retrieval plus equivalent-query consistency."""
import argparse
import json
import time

import torch
import torch.nn.functional as F

import ternary_qat
from diagnose_reasoning import CURRENT, OUT
from domain import evaluate, listwise_loss, replay_groups
from native import embed_all
from prep.compositional import CACHE, CHALLENGE
from prep.domain import CACHE as DOMAIN
from prep.native import ROOT, digest
from prep.round2 import encode_rows
from reasoning import fixtures, geometry_loss, load_model
from reasoning_calibrate import language_mask
from round2 import encode, take
from tokenization import load_tokenizer


def objective(v, ids, control=False):
    if control:
        return F.softplus((.035-(v[:,0]*(v[:,2]-v[:,3])).sum(-1))*32).mean()
    q=v[:,:2].reshape(-1,v.shape[-1]);d=v[:,2:].reshape(-1,v.shape[-1])
    relevant=ids[:,0].repeat_interleave(2)[:,None]==ids.flatten()[None,:]
    scores=q@d.T*32
    rank=(torch.logsumexp(scores,1)-torch.logsumexp(scores.masked_fill(~relevant,float('-inf')),1)).mean()
    own=torch.einsum('bqd,bkd->bqk',v[:,:2],v[:,2:])
    worst=own[:,:,0]-own[:,:,1:].max(-1).values
    return rank+F.softplus((.02-worst)*32).mean()+.5*(1-(v[:,0]*v[:,1]).sum(-1)).mean()


@torch.no_grad()
def logic_metrics(model,data,device):
    model.eval();original=model.embedding.weight.detach().clone();ternary_qat.int4_quantize_embedding_(model)
    try:
        vectors=embed_all(model,data,device);result={}
        for group,name in enumerate(['negation','numbers']):
            mask=data['group']==group;v=vectors[mask];ids=data['semantic_ids'][mask]
            q=v[:,:2];d=v[:,2:];scores=torch.einsum('bqd,bkd->bqk',q,d)
            wins=scores[:,:,0]-scores[:,:,1:].max(-1).values>1e-6
            global_scores=q.reshape(-1,384)@d.reshape(-1,384).T
            relevant=ids[:,0].repeat_interleave(2)[:,None]==ids.flatten()[None,:]
            gap=global_scores.masked_fill(~relevant,-float('inf')).max(-1).values-global_scores.masked_fill(relevant,-float('inf')).max(-1).values
            result[name]=dict(count=len(v),exact=float(wins.float().mean()),both_views=float(wins.all(1).float().mean()),full_top1=float((gap>1e-6).float().mean()))
        return result
    finally:model.embedding.weight.copy_(original)


def retention_gates(m,b,logic,base_logic):
    groups={
        'english':all(abs(m[f'r{i}_en_ndcg']-b[f'r{i}_en_ndcg'])<1e-6 for i in [2,3,4]) and abs(m['sts_en']-b['sts_en'])<1e-6,
        'persian_ranking':all(m[f'r{i}_fa_ndcg']>=b[f'r{i}_fa_ndcg']-.01 for i in [2,3,4]),
        'persian_sts':m['sts_fa']>=b['sts_fa']-.002,
        'docs_emoji':all(m[p+t+s]>=b[p+t+s]-1e-6 for p in ['domain_','previous_','reasoning_'] for t in ['docs','emoji'] for s in ['_ndcg','_top1']),
        'prior_logic':all(m[k]>=b[k]-1e-6 for k in ['negation','numbers','reasoning_fa_negation_top1','reasoning_fa_numbers_top1']),
        'new_logic':all(logic[c][k]>=base_logic[c][k]-1e-6 for c in ['negation','numbers'] for k in ['both_views','full_top1'])}
    return groups


def main(control=False,smoke=False):
    torch.manual_seed(829);torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu'
    name='mini-fa-en-compositional'+('-control' if control else '')+('-smoke' if smoke else '')
    run=ROOT/'training/distill/runs'/name
    if run.exists():raise FileExistsError('Preserve experiments')
    manifest=json.loads((CACHE/'manifest.json').read_text())
    assert digest(CURRENT)==manifest['source_sha256'] and digest(CHALLENGE)==manifest['challenge_sha256']
    for file,sha in manifest['files'].items():assert digest(CACHE/file)==sha
    source=torch.load(CURRENT,map_location='cpu',weights_only=False);tk=load_tokenizer(source['tokenizer'])
    model=load_model(source,device,True);model.requires_grad_(False);model.embedding.weight.requires_grad_(True)
    mask=language_mask(tk.get_vocab()).to(device);model.embedding.weight.register_hook(lambda g:g*mask[:,None])
    teacher=load_model(source,device).eval();reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    data=torch.load(CACHE/'data.pt',weights_only=False);domain=torch.load(DOMAIN/'data.pt',weights_only=False)
    old=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False)
    texts=json.loads((ROOT/'training/distill/cache/mini-fa-en-round2/texts.json').read_text())['replay_texts'];opus=json.loads((ROOT/'training/distill/cache/mini-fa-en-32k/pairs.json').read_text())['train'];fa=replay_groups(texts,opus)[1]
    cases=fixtures();prior=json.loads((ROOT/'eval/quality/corpus/persian-reasoning-hard.json').read_text())
    cases.update({'reasoning_'+k:f for k,f in prior['suites'].items()})
    tokens={k:encode_rows(f['corpus']+f['queries'],['text'],tk) for k,f in cases.items()}
    baseline=evaluate(model,cases,tokens,old['sts_val'],device);base_logic=logic_metrics(model,data['val'],device)
    run.mkdir(parents=True);recipe=dict(seed=829,epochs=4,steps=250,lr=.00018,control=control,source_sha256=digest(CURRENT),manifest=manifest,
        training='Persian-script/digit embedding rows only, shared body and non-Persian rows frozen from step zero, int4 STE. No transplant, aliases or runtime arithmetic.',
        loss='Binary contrast control' if control else 'Multi-positive full-batch retrieval + nearest-answer margin + equivalent-query consistency',
        retention='4x current domain geometry, 2x current Persian replay geometry, domain listwise; exact English weight invariants',
        selection='Eligible first, then mean validation exact-both and full-corpus top1 across FA negation/numbers. Diagnostic only if none eligible; no fresh inference before selection.')
    (run/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');(run/'baseline.json').write_text(json.dumps(dict(metrics=baseline,logic=base_logic),indent=2)+'\n')
    groups=[torch.where(data['train']['group']==g)[0] for g in [0,1]];optimizer=torch.optim.AdamW([model.embedding.weight],lr=recipe['lr'],weight_decay=0)
    best=(-1,-1.);history=[]
    for epoch in range(1,(1 if smoke else recipe['epochs'])+1):
        start=time.monotonic();model.train()
        for step in range(1 if smoke else recipe['steps']):
            ids=torch.cat([g[torch.randint(len(g),(12,))] for g in groups]);batch=take(data['train'],ids);v=encode(model,batch,device)
            logic=objective(v,batch['semantic_ids'].to(device),control)
            task=['docs','emoji'][step%2];b=take(domain[task+'_train'],torch.randint(len(domain[task+'_train']['input_ids']),(16,)))
            q=encode(model,b,device)[:,0];d=encode(model,domain[task+'_corpus'],device)[:,0];rank=listwise_loss(q,d,b['relevant'].to(device))
            with torch.no_grad():tq=encode(teacher,b,device)[:,0];td=encode(teacher,domain[task+'_corpus'],device)[:,0]
            retention=4*geometry_loss(torch.cat([q,d]),torch.cat([tq,td]))
            b=take(old['replay'],fa[torch.randint(len(fa),(32,))]);rv=encode(model,b,device)
            with torch.no_grad():tv=encode(teacher,b,device)
            retention+=2*geometry_loss(rv,tv);loss=logic+rank+retention
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            progress=(epoch-1)*recipe['steps']+step;optimizer.param_groups[0]['lr']=recipe['lr']*min(1,(progress+1)/30)*(1-progress/(recipe['epochs']*recipe['steps']))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([model.embedding.weight],1);optimizer.step()
            if (step+1)%50==0 or smoke:print(dict(epoch=epoch,step=step+1,loss=float(loss.detach()),logic=float(logic.detach()),retention=float(retention.detach())),flush=True)
        for k,v in model.state_dict().items():
            assert torch.equal(v.detach().cpu()[~mask.cpu()] if k=='embedding.weight' else v.detach().cpu(),reference[k][~mask.cpu()] if k=='embedding.weight' else reference[k]),k
        metrics=evaluate(model,cases,tokens,old['sts_val'],device);lm=logic_metrics(model,data['val'],device);gates=retention_gates(metrics,baseline,lm,base_logic)
        score=sum(lm[c][k] for c in lm for k in ['both_views','full_top1'])/4;key=(int(all(gates.values())),score)
        record=dict(epoch=epoch,seconds=time.monotonic()-start,metrics=metrics,logic=lm,gates=gates,score=score);history.append(record);print('EPOCH',json.dumps(record),flush=True)
        ckpt=dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config={**source['config'],'run_name':name},tokenizer=source['tokenizer'],epoch=epoch,validation=record,recipe=recipe)
        if key>best:best=key;torch.save(ckpt,run/'selected.pt')
        if all(gates.values()):torch.save(ckpt,run/'eligible.pt')
        torch.save(ckpt,run/f'epoch{epoch}.pt');(run/'metrics.json').write_text(json.dumps(history,indent=2)+'\n')
    checkpoint=run/'selected.pt';ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    (run/'selection.json').write_text(json.dumps(dict(checkpoint=str(checkpoint.relative_to(ROOT)),sha256=digest(checkpoint),validation=ckpt['validation'],release_ready=False,challenge_sha256=digest(CHALLENGE)),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.control,a.smoke)
