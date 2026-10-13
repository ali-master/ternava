"""Validation-only English-safe transplant after shared-weight retention failed."""
import json
import unicodedata

import torch

from native import persian_rows
from prep.native import ROOT, digest
from prep.reasoning import CACHE, SOURCE
from prep.round2 import encode_rows
from reasoning import EN_SOURCE, assess, eligible, fixtures, load_model
from tokenization import load_tokenizer


def language_mask(vocab):
    mask=persian_rows(vocab)
    for token,i in vocab.items():
        chars=token.removeprefix('##')
        if chars and all('\u0600'<=c<='\u06ff' and unicodedata.category(c).startswith(('L','N')) for c in chars):mask[i]=True
    return mask


def main():
    torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu'
    run=ROOT/'training/distill/runs/mini-fa-en-reasoning'
    if (run/'calibration.json').exists():raise FileExistsError('Preserve consumed calibration')
    assert (run/'checkpoint_ep6.pt').exists()
    candidate=torch.load(run/'diagnostic-best.pt',map_location='cpu',weights_only=False)
    domain=torch.load(SOURCE,map_location='cpu',weights_only=False);english=torch.load(EN_SOURCE,map_location='cpu',weights_only=False)
    tk=load_tokenizer(english['tokenizer']);mask=language_mask(tk.get_vocab())
    cases=fixtures();tokens={k:encode_rows(f['corpus']+f['queries'],['text'],tk) for k,f in cases.items()}
    old=torch.load(ROOT/'training/distill/cache/mini-fa-en-round2/data.pt',weights_only=False);logic=torch.load(CACHE/'data.pt',weights_only=False)['val']
    baseline=json.loads((run/'baseline.json').read_text());enbase=json.loads((run/'english-baseline.json').read_text());results=[]
    best=(-1,-float('inf'))
    for alpha in [0.,.5,1.]:
        state={k:v.clone() for k,v in english['model_state'].items()}
        state['embedding.weight'][mask]=torch.lerp(domain['model_state']['embedding.weight'][mask],candidate['model_state']['embedding.weight'][mask],alpha)
        model=load_model({**candidate,'model_state':state},device,True).eval()
        m=assess(model,cases,tokens,old['sts_val'],logic,device);m.update(alpha=alpha,eligible=eligible(m,baseline,enbase))
        assert all(abs(m[f'r{i}_en_ndcg']-enbase[f'r{i}_en_ndcg'])<1e-6 for i in [2,3,4]),'English fixture includes modified tokens or shared weights drifted'
        results.append(m);print('CALIBRATION',m,flush=True)
        key=(int(m['eligible']),m['selection_score'])
        if key>best:
            best=key;torch.save({**candidate,'model_state':state,'validation':m,'calibration':dict(alpha=alpha,mask_rows=int(mask.sum()),english_sha256=digest(EN_SOURCE),domain_sha256=digest(SOURCE),candidate_sha256=digest(run/'diagnostic-best.pt'),policy='Restore all shared and non-Persian weights to round3; interpolate only Persian-script/digit rows. Validation-only alpha 0/.5/1.')},run/'english-safe.pt')
    (run/'calibration.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
