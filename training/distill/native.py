"""Round-three native Persian continuation; no inference-time language branch."""
import argparse
import json
import time
import unicodedata

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from model import StudentEncoder
from prep.native import CACHE, ROOT, SOURCE, digest
from prep.round2 import encode_rows
from round2 import encode, take, triplet_loss, cosent_loss
from tokenization import load_tokenizer
import ternary_qat


def persian_rows(vocab):
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    for token, index in vocab.items():
        letters = [c for c in token if unicodedata.category(c).startswith('L')]
        mask[index] = bool(letters) and all('\u0600' <= c <= '\u06ff' for c in letters)
    mask[0] = False
    return mask


def int4_ste(module, args, output):
    if not module.training: return output
    scale = (output.detach().abs().amax(-1,keepdim=True) / 7).clamp_min(1e-8)
    quantized = (output/scale).round().clamp(-7,7)*scale
    return output + (quantized-output).detach()


def retrieval_loss(vectors, same_positive):
    scores = vectors[:,0] @ vectors[:,1].T * 12
    # A batch may contain several questions for the same answer window.
    return (torch.logsumexp(scores,dim=1) -
            torch.logsumexp(scores.masked_fill(~same_positive,float('-inf')),dim=1)).mean()


@torch.no_grad()
def embed_all(model, data, device):
    return torch.cat([encode(model,take(data,slice(i,i+48)),device).cpu()
                      for i in range(0,len(data['input_ids']),48)])


@torch.no_grad()
def evaluate(model, data, qa, qa_tokens, sts, device):
    model.eval(); original = model.embedding.weight.detach().clone()
    ternary_qat.int4_quantize_embedding_(model)
    try:
        v = embed_all(model,qa_tokens,device)[:,0]; nc=len(qa['corpus'])
        scores = (v[nc:]@v[:nc].T).numpy(); reciprocal=[]; ndcg=[]; top=[]
        for q,row in zip(qa['queries'],scores):
            ranked = np.argsort(-row,kind='stable'); relevant={int(k) for k in q['relevant']}
            hits = [i+1 for i,d in enumerate(ranked) if d in relevant]
            reciprocal.append(1/hits[0]); top.append(hits[0]==1)
            ndcg.append(sum(1/np.log2(r+1) for r in hits if r<=10)/sum(1/np.log2(r+1) for r in range(1,min(len(relevant),10)+1)))
        qq=embed_all(model,data['qqp_val'],device); nli=embed_all(model,data['nli_val'],device)
        s=embed_all(model,sts,device); sim=(s[:,0]*s[:,1]).sum(-1)
        result=dict(qa_mrr=float(np.mean(reciprocal)),qa_ndcg=float(np.mean(ndcg)),qa_top1=float(np.mean(top)),
                    qqp_spearman=float(spearmanr((qq[:,0]*qq[:,1]).sum(-1),data['qqp_val']['label']).statistic),
                    nli_contrast=float(((nli[:,0]*(nli[:,1]-nli[:,2])).sum(-1)>1e-6).float().mean()))
        for tag,lang in [(0,'en'),(1,'fa')]:
            keep=sts['lang']==tag; result['sts_'+lang]=float(spearmanr(sim[keep],sts['score'][keep]).statistic)
        result['selection_score']=result['qa_ndcg']+.1*result['qqp_spearman']+.1*result['nli_contrast']
        assert all(np.isfinite(x) for x in result.values())
        return result
    finally: model.embedding.weight.copy_(original)


def train(smoke=False):
    torch.manual_seed(531);np.random.seed(531);torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    manifest=json.loads((CACHE/'manifest.json').read_text())
    assert digest(SOURCE)==manifest['source_sha256']
    for name,sha in manifest['files'].items(): assert digest(CACHE/name)==sha
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    tk=load_tokenizer(source['tokenizer']);rows=persian_rows(tk.get_vocab())
    data=torch.load(CACHE/'data.pt',weights_only=False)
    old=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False)
    qa=json.loads((CACHE/'qa-val.json').read_text())
    qa_tokens=encode_rows(qa['corpus']+qa['queries'],['text'],tk)
    cfg={**source['config'],'run_name':'mini-fa-en-round3-native','seed':531,'epochs':4,'lr':.0004,'dropout':0.}
    run=ROOT/'training/distill/runs'/(cfg['run_name']+('-smoke' if smoke else ''))
    if run.exists(): raise FileExistsError('Preserve runs; use a new directory for an explicitly revised experiment')
    run.mkdir(parents=True)
    args={k:cfg[k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']}
    model=StudentEncoder(**args);model.load_state_dict(source['model_state'])
    ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model.to(device).requires_grad_(False)
    reference={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.embedding.weight.requires_grad_(True)
    row_mask=rows[:,None].to(device)
    model.embedding.weight.register_hook(lambda grad: grad*row_mask)
    model.embedding.register_forward_hook(int4_ste)
    anchor=StudentEncoder(**args);anchor.load_state_dict(source['model_state'])
    ternary_qat.swap(anchor);ternary_qat.set_lambda(anchor,1);ternary_qat.int4_quantize_embedding_(anchor)
    anchor.to(device).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([model.embedding.weight],lr=cfg['lr'],weight_decay=0)
    replay_ids=torch.arange(50000,len(old['replay']['input_ids']))
    baseline=evaluate(model,data,qa,qa_tokens,old['sts_val'],device)
    (run/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    recipe=dict(steps_per_epoch=400,qa_batch=32,nli_batch=16,qqp_batch=24,replay_batch=24,
                qa_weight=1,nli_weight=.5,qqp_weight=.1,replay_weight=.2,
                trainable_rows=int(rows.sum()),selection='Native validation only, int4; fa STS at least 97% baseline; English tensors invariant. All old hard fixtures are regression tests. No fresh-test selection.',
                fresh_demo_sha256=digest(ROOT/'eval/quality/corpus/persian-native-demos.json'))
    (run/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    print('BASELINE',baseline,'trainable_rows',int(rows.sum()),flush=True)
    best=-float('inf');best_epoch=None;diagnostic=-float('inf')
    for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
        start=time.monotonic();model.train();losses=[]
        for step in range(1 if smoke else recipe['steps_per_epoch']):
            sample=lambda key,n:torch.randint(len(data[key]['input_ids']),(n,))
            ix=sample('qa_train',32)
            batch={k:v[ix] for k,v in data['qa_train'].items() if k!='same_positive'}
            q=encode(model,batch,device)
            ranking=retrieval_loss(q,data['qa_train']['same_positive'][ix][:,ix].to(device))
            n=encode(model,take(data['nli_train'],sample('nli_train',16)),device)
            qb=take(data['qqp_train'],sample('qqp_train',24));qq=encode(model,qb,device)
            replay=take(old['replay'],replay_ids[torch.randint(len(replay_ids),(24,))])
            r=encode(model,replay,device)
            with torch.no_grad(): target=encode(anchor,replay,device)
            retain=(1-(r*target).sum(-1)).mean()
            loss=ranking+.5*triplet_loss(n)+.1*cosent_loss(qq,qb['label'].to(device))+.2*retain
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            optimizer.param_groups[0]['lr']=cfg['lr']*min(1,((epoch-1)*400+step+1)/50)*(1-((epoch-1)*400+step)/1600)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([model.embedding.weight],1);optimizer.step()
            losses.append(float(loss.detach()))
            if (step+1)%100==0 or smoke: print(f'epoch={epoch} step={step+1} loss={losses[-1]:.4f} qa={float(ranking.detach()):.4f}',flush=True)
        for name,tensor in model.state_dict().items():
            actual=tensor.detach().cpu(); expected=reference[name]
            assert torch.equal(actual[~rows],expected[~rows]) if name=='embedding.weight' else torch.equal(actual,expected),name
        metrics=evaluate(model,data,qa,qa_tokens,old['sts_val'],device) if not smoke else baseline.copy()
        eligible=metrics['sts_fa']>=baseline['sts_fa']*.97 and abs(metrics['sts_en']-baseline['sts_en'])<1e-7
        metrics.update(epoch=epoch,eligible=eligible,seconds=time.monotonic()-start,loss=float(np.mean(losses)))
        print('EPOCH',metrics,flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        ckpt=dict(epoch=epoch,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,
                  tokenizer=source['tokenizer'],data_manifest=manifest,source_checkpoint_sha256=digest(SOURCE),validation=metrics,recipe=recipe)
        if metrics['selection_score']>diagnostic:
            diagnostic=metrics['selection_score'];torch.save(ckpt,run/'diagnostic-best.pt')
        if eligible and metrics['selection_score']>best:
            best=metrics['selection_score'];best_epoch=epoch;torch.save(ckpt,run/'best.pt')
        if epoch==cfg['epochs'] or smoke:torch.save(ckpt,run/f'checkpoint_ep{epoch}.pt')
    print(f'VERDICT: {"WIN" if best>baseline["selection_score"] else "REGRESSION"} | validation={best} | baseline={baseline["selection_score"]} | epoch={best_epoch}',flush=True)


def calibrate():
    """Validation-only fallback if the fixed run overspecializes QA embeddings."""
    torch.set_num_threads(2)
    run=ROOT/'training/distill/runs/mini-fa-en-round3-native'
    if (run/'best.pt').exists() or (run/'calibrated.pt').exists():
        raise FileExistsError('Calibration is only for a run with no eligible checkpoint')
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    baseline=json.loads((run/'baseline.json').read_text())
    ckpt=torch.load(run/'diagnostic-best.pt',map_location='cpu',weights_only=False)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    tk=load_tokenizer(ckpt['tokenizer']);mask=persian_rows(tk.get_vocab())
    qa=json.loads((CACHE/'qa-val.json').read_text());tokens=encode_rows(qa['corpus']+qa['queries'],['text'],tk)
    data=torch.load(CACHE/'data.pt',weights_only=False)
    sts=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False)['sts_val']
    args={k:ckpt['config'][k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']}
    best=baseline['selection_score'];results=[]
    for alpha in [.25,.5,.75]:
        model=StudentEncoder(**args);model.load_state_dict(ckpt['model_state'])
        with torch.no_grad():
            model.embedding.weight[mask]=torch.lerp(source['model_state']['embedding.weight'][mask],ckpt['model_state']['embedding.weight'][mask],alpha)
        ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model.to(device)
        metrics=evaluate(model,data,qa,tokens,sts,device)
        metrics.update(alpha=alpha,eligible=metrics['sts_fa']>=baseline['sts_fa']*.97 and abs(metrics['sts_en']-baseline['sts_en'])<1e-7)
        results.append(metrics);print('CALIBRATION',metrics,flush=True)
        if metrics['eligible'] and metrics['selection_score']>best:
            best=metrics['selection_score']
            output={**ckpt,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'validation':metrics,
                    'calibration':dict(alpha=alpha,diagnostic_checkpoint_sha256=digest(run/'diagnostic-best.pt'),
                        reason='No eligible checkpoint in fixed run; only native validation used to choose among .25/.5/.75 interpolation of Persian rows')}
            torch.save(output,run/'calibrated.pt')
    (run/'calibration.json').write_text(json.dumps(results,indent=2)+'\n')
    if not (run/'calibrated.pt').exists():raise RuntimeError('No validated continuation; do not promote')
    print(f'VERDICT: WIN | validation={best:.6f} | baseline={baseline["selection_score"]:.6f} | test verdict pending',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--smoke',action='store_true');group.add_argument('--calibrate',action='store_true')
    args=parser.parse_args()
    calibrate() if args.calibrate else train(args.smoke)
