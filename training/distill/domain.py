"""Fixed-size technical/colloquial Persian QAT with English replay and logic gates."""
import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from model import StudentEncoder
from native import persian_rows, int4_ste, embed_all
from prep.domain import CACHE, CHALLENGE
from prep.native import ROOT, SOURCE, digest
from prep.round2 import encode_rows, normalize
from round2 import encode, take, cosent_loss
from round3 import ranking_loss
from round4 import contrast_loss
from tokenization import load_tokenizer
import ternary_qat


def listwise_loss(queries, documents, relevant):
    scores=queries@documents.T*16
    return (torch.logsumexp(scores,dim=1)-torch.logsumexp(scores.masked_fill(~relevant,float('-inf')),dim=1)).mean()


def replay_groups(texts, pairs, opus_count=60000):
    languages={row[lang]:lang for row in pairs for lang in ['en','fa']}
    # The first 60k rows were shuffled; only the appended AllNLI tail is all English.
    tags=[languages[t] if i<opus_count else 'en' for i,t in enumerate(texts)]
    return [torch.tensor([i for i,tag in enumerate(tags) if tag==lang]) for lang in ['en','fa']]


def filter_triplets(data, texts, forbidden, paired=False):
    good=[]
    for i,row in enumerate(texts):
        collision=torch.equal(data['input_ids'][i,1],data['input_ids'][i,2])
        good.append(not collision and not any(normalize(row[k]) in forbidden for k in ['anchor','positive','negative']))
    if paired:
        assert len(good)%2==0
        for i in range(0,len(good),2):good[i]=good[i+1]=good[i] and good[i+1]
    indices=torch.tensor([i for i,ok in enumerate(good) if ok])
    return take(data,indices)


def validation_fixtures():
    fixtures=json.loads((CACHE/'validation.json').read_text())
    for i in [2,3,4]:
        f=json.loads((ROOT/f'eval/quality/corpus/bilingual-hard-round{i}.json').read_text())
        for lang in ['fa','en']:
            fixtures[f'r{i}_{lang}']={key:[r for r in f[key] if r['lang']==lang] for key in ['corpus','queries']}
    # Previous domain challenge is now explicitly development data, never a fresh test.
    old=json.loads((ROOT/'eval/quality/corpus/persian-native-demos.json').read_text())
    for key in ['docs','emoji','odyssey']:
        fixtures['previous_'+key]=dict(corpus=old['corpora'][key],queries=old['queries'][key])
    return fixtures


@torch.no_grad()
def evaluate(model, fixtures, tokens, sts, device):
    model.eval();original=model.embedding.weight.detach().clone();ternary_qat.int4_quantize_embedding_(model)
    try:
        metrics={};contrasts={'negation':[],'numbers':[]}
        for name,f in fixtures.items():
            v=embed_all(model,tokens[name],device)[:,0];n=len(f['corpus'])
            scores=(v[n:]@v[:n].T).numpy();lookup={d.get('id',str(i)):i for i,d in enumerate(f['corpus'])}
            ndcg=[];top=[]
            for q,row in zip(f['queries'],scores):
                relevant={lookup[k] for k in q['relevant']}
                order=np.argsort(-row,kind='stable');hits=[i+1 for i,d in enumerate(order) if d in relevant]
                top.append(hits[0]==1)
                ndcg.append(sum(1/np.log2(r+1) for r in hits if r<=10)/sum(1/np.log2(r+1) for r in range(1,min(len(hits),10)+1)))
                if name.endswith('_fa') and q.get('category') in contrasts:
                    contrasts[q['category']].append(max(row[j] for j in relevant)-row[lookup[q['contrast']]]>1e-6)
            metrics[name+'_ndcg']=float(np.mean(ndcg));metrics[name+'_top1']=float(np.mean(top))
        for name,values in contrasts.items():metrics[name]=float(np.mean(values))
        v=embed_all(model,sts,device);sim=(v[:,0]*v[:,1]).sum(-1)
        for tag,lang in [(0,'en'),(1,'fa')]:
            keep=sts['lang']==tag;metrics['sts_'+lang]=float(spearmanr(sim[keep],sts['score'][keep]).statistic)
        metrics['selection_score']=.5*(metrics['docs_ndcg']+metrics['emoji_ndcg'])+.15*min(metrics['negation'],metrics['numbers'])
        return metrics
    finally:model.embedding.weight.copy_(original)


def eligible(m,b):
    return (m['sts_en']>=b['sts_en']-.002 and m['sts_fa']>=b['sts_fa']-.01
            and all(m[k]>=b[k] for k in ['negation','numbers'])
            and all(m[f'r{i}_{l}_ndcg']>=b[f'r{i}_{l}_ndcg']-.02 for i in [2,3,4] for l in ['fa','en'])
            and all(m['previous_'+k+'_top1']>=b['previous_'+k+'_top1'] for k in ['docs','emoji','odyssey']))


def train(smoke=False, retention_repair=False):
    torch.manual_seed(613);np.random.seed(613);torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    manifest=json.loads((CACHE/'manifest.json').read_text());assert digest(SOURCE)==manifest['source_sha256']
    assert digest(CHALLENGE)==manifest['challenge_sha256']
    for name,sha in manifest['files'].items():assert digest(CACHE/name)==sha
    source=torch.load(SOURCE,map_location='cpu',weights_only=False);tk=load_tokenizer(source['tokenizer'])
    domain=torch.load(CACHE/'data.pt',weights_only=False)
    old=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False)
    replay_texts=json.loads((ROOT/'training/distill/cache/mini-fa-en-round2/texts.json').read_text())['replay_texts']
    opus=json.loads((ROOT/'training/distill/cache/mini-fa-en-32k/pairs.json').read_text())['train']
    replay_indices=replay_groups(replay_texts,opus)
    assert [len(g) for g in replay_indices]==[50000,30000]
    r3path=ROOT/'training/distill/cache/mini-fa-en-round3';r4path=ROOT/'training/distill/cache/mini-fa-en-round4'
    fresh=json.loads(CHALLENGE.read_text());forbidden={normalize(r['text']) for name,f in fresh['suites'].items() for r in f['queries']+(f['corpus'] if name in ['logic','unseen_corpus'] else [])}
    r3=filter_triplets(torch.load(r3path/'data.pt',weights_only=False)['train'],json.loads((r3path/'texts.json').read_text())['train'],forbidden)
    logic=filter_triplets(torch.load(r4path/'data.pt',weights_only=False)['train'],json.loads((r4path/'texts.json').read_text())['train'],forbidden,paired=True)
    groups=[torch.where(logic['group'][::2]==g)[0]*2 for g in [0,1,3,4]]
    previous_groups=[torch.where(r3['group']==g)[0] for g in range(7)]
    sts_groups=[torch.where(old['sts_train']['lang']==g)[0] for g in [0,1]]
    fixtures=validation_fixtures();tokens={k:encode_rows(f['corpus']+f['queries'],['text'],tk) for k,f in fixtures.items()}
    steps,epochs,embedding_lr,body_lr,fa_retention,old_rank_weight=(200,3,.00006,.000001,20,1.) if retention_repair else (250,6,.00012,.000003,2,.3)
    cfg={**source['config'],'run_name':'mini-fa-en-domain'+('-retention' if retention_repair else ''),'seed':613,'epochs':epochs,'lr':embedding_lr,'dropout':0.}
    run=ROOT/'training/distill/runs'/(cfg['run_name']+('-smoke' if smoke else ''))
    if run.exists():raise FileExistsError('Preserve all runs')
    run.mkdir(parents=True)
    args={k:cfg[k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']}
    model=StudentEncoder(**args);model.load_state_dict(source['model_state']);ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model.to(device)
    mask=persian_rows(tk.get_vocab()).to(device);model.embedding.weight.register_hook(lambda g:g*mask[:,None]);model.embedding.register_forward_hook(int4_ste)
    anchor=StudentEncoder(**args);anchor.load_state_dict(source['model_state']);ternary_qat.swap(anchor);ternary_qat.set_lambda(anchor,1);ternary_qat.int4_quantize_embedding_(anchor);anchor.to(device).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([{'params':[model.embedding.weight],'lr':embedding_lr},
                                {'params':[p for n,p in model.named_parameters() if n!='embedding.weight'],'lr':body_lr}],weight_decay=0)
    recipe=dict(steps_per_epoch=steps,epochs=epochs,embedding_lr=embedding_lr,backbone_lr=body_lr,domain_batch=24,paired_logic_batch=24,
                previous_triplets=28,replay_en=24,replay_fa=16,sts_en=16,sts_fa=16,
                loss=f'1 domain listwise + 3 logic margin + {old_rank_weight} old listwise + 30 English cosine retention + {fa_retention} Persian cosine retention + .05 bilingual CoSENT',
                calibration='If no eligible checkpoint, validation-only interpolation of diagnostic-best at .25/.5/.75; same gates; no test tuning',
                selection='Mean domain NDCG plus .15 worst FA logic contrast. English validation STS loss <=.002; FA <=.01; all old rank losses <=.02 absolute; aggregate FA negation/numbers and previous demo top1 cannot regress.',
                triplets_after_filter=len(r3['input_ids']),paired_logic_after_filter=len(logic['input_ids']))
    (run/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    baseline=evaluate(model,fixtures,tokens,old['sts_val'],device)
    (run/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');print('BASELINE',baseline,flush=True)
    best=-float('inf');diagnostic=-float('inf');best_epoch=None
    for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
        start=time.monotonic();model.train();losses=[]
        for step in range(1 if smoke else steps):
            task=['docs','emoji'][step%2];batch=take(domain[task+'_train'],torch.randint(len(domain[task+'_train']['input_ids']),(24,)))
            q=encode(model,batch,device)[:,0];docs=encode(model,domain[task+'_corpus'],device)[:,0]
            rank=listwise_loss(q,docs,batch['relevant'].to(device))
            starts=torch.cat([g[torch.randint(len(g),(3,))] for g in groups]);ix=torch.stack([starts,starts+1],1).flatten()
            lv=encode(model,take(logic,ix),device);contrast=contrast_loss(lv)
            previous=take(r3,torch.cat([g[torch.randint(len(g),(4,))] for g in previous_groups]));pv=encode(model,previous,device)
            oldrank=ranking_loss(pv,previous['topic'].to(device),previous['doc_ids'].to(device),previous['group'].to(device))
            replay=take(old['replay'],torch.cat([g[torch.randint(len(g),(n,))] for g,n in zip(replay_indices,[24,16])]))
            rv=encode(model,replay,device)
            with torch.no_grad():target=encode(anchor,replay,device)
            delta=1-(rv*target).sum(-1);retention=30*delta[:24].mean()+fa_retention*delta[24:].mean()
            scored=take(old['sts_train'],torch.cat([g[torch.randint(len(g),(16,))] for g in sts_groups]));sv=encode(model,scored,device)
            sts=(cosent_loss(sv[:16],scored['score'][:16].to(device))+cosent_loss(sv[16:],scored['score'][16:].to(device)))/2
            loss=rank+3*contrast+old_rank_weight*oldrank+retention+.05*sts
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            progress=(epoch-1)*steps+step;factor=min(1,(progress+1)/50)*(1-progress/(steps*epochs))
            for group,lr in zip(optimizer.param_groups,[embedding_lr,body_lr]):group['lr']=lr*factor
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%50==0 or smoke:print(f'epoch={epoch} step={step+1} loss={losses[-1]:.4f} domain={float(rank.detach()):.4f} logic={float(contrast.detach()):.4f} retention={float(retention.detach()):.4f}',flush=True)
        assert torch.equal(model.embedding.weight.detach().cpu()[~mask.cpu()],source['model_state']['embedding.weight'][~mask.cpu()])
        metrics=evaluate(model,fixtures,tokens,old['sts_val'],device) if not smoke else baseline.copy()
        metrics.update(epoch=epoch,eligible=eligible(metrics,baseline),seconds=time.monotonic()-start,loss=float(np.mean(losses)))
        print('EPOCH',metrics,flush=True)
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        ckpt=dict(epoch=epoch,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,tokenizer=source['tokenizer'],data_manifest=manifest,source_checkpoint_sha256=digest(SOURCE),validation=metrics,recipe=recipe)
        if metrics['selection_score']>diagnostic:diagnostic=metrics['selection_score'];torch.save(ckpt,run/'diagnostic-best.pt')
        if metrics['eligible'] and metrics['selection_score']>best:best=metrics['selection_score'];best_epoch=epoch;torch.save(ckpt,run/'best.pt')
        if epoch==cfg['epochs'] or smoke:torch.save(ckpt,run/f'checkpoint_ep{epoch}.pt')
    score=best if best_epoch is not None else diagnostic
    print(f'VERDICT: {"WIN" if best>baseline["selection_score"] else "REGRESSION"} | score={score:.6f} | baseline={baseline["selection_score"]:.6f} | delta={score-baseline["selection_score"]:.6f} | eligible_epoch={best_epoch}',flush=True)


def calibrate(retention_repair=False):
    """Use only the preregistered validation fallback, never fresh test scores."""
    torch.set_num_threads(2)
    run=ROOT/'training/distill/runs'/('mini-fa-en-domain'+('-retention' if retention_repair else ''))
    if any((run/p).exists() for p in ['best.pt','calibrated.pt','calibration.json']):
        raise FileExistsError('Only calibrate a completed run without eligible checkpoints')
    assert (run/f'checkpoint_ep{3 if retention_repair else 6}.pt').exists()
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    baseline=json.loads((run/'baseline.json').read_text())
    checkpoint=torch.load(run/'diagnostic-best.pt',map_location='cpu',weights_only=False)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    tk=load_tokenizer(checkpoint['tokenizer']);cases=validation_fixtures()
    tokens={k:encode_rows(f['corpus']+f['queries'],['text'],tk) for k,f in cases.items()}
    sts=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False)['sts_val']
    args={k:checkpoint['config'][k] for k in ['vocab_size','d_model','n_layers','n_heads','ffn_dim','output_dim','dropout','position_embeddings']}
    best=baseline['selection_score'];results=[]
    for alpha in [.25,.5,.75]:
        model=StudentEncoder(**args)
        model.load_state_dict({k:torch.lerp(source['model_state'][k],v,alpha) for k,v in checkpoint['model_state'].items()})
        ternary_qat.swap(model);ternary_qat.set_lambda(model,1);model.to(device)
        metrics=evaluate(model,cases,tokens,sts,device);metrics.update(alpha=alpha,eligible=eligible(metrics,baseline))
        results.append(metrics);print('CALIBRATION',metrics,flush=True)
        if metrics['eligible'] and metrics['selection_score']>best:
            best=metrics['selection_score']
            torch.save({**checkpoint,'model_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'validation':metrics,
                        'calibration':dict(alpha=alpha,diagnostic_sha256=digest(run/'diagnostic-best.pt'),selection='Validation-only fixed .25/.5/.75 interpolation')},run/'calibrated.pt')
    (run/'calibration.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'VERDICT: {"WIN" if best>baseline["selection_score"] else "REGRESSION"} | validation={best:.6f} | baseline={baseline["selection_score"]:.6f} | test verdict pending',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--calibrate',action='store_true');p.add_argument('--retention-repair',action='store_true')
    args=p.parse_args();calibrate(args.retention_repair) if args.calibrate else train(args.smoke,args.retention_repair)
