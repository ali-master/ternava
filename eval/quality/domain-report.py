"""Summarize frozen domain runs with explicit, non-averaged release gates."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/pilot-fa-en-domain'


def main():
    reports={name:json.loads((OUT/f'eval-{name}.json').read_text()) for name in ['round3','native','candidate']}
    selection=json.loads((OUT/'selection.json').read_text())
    sts={name:json.loads((ROOT/f'dist/{folder}/sts-candidate.json').read_text()) for name,folder in
         [('round3','pilot-fa-en-v3'),('native','pilot-fa-en-v3-native'),('candidate','pilot-fa-en-domain')]}
    for name,r in reports.items():
        assert r['inputs']==reports['round3']['inputs']
        assert r['tokenizer_sha256']==reports['round3']['tokenizer_sha256']
        assert sts[name]['dataset_sha256']==sts['round3']['dataset_sha256']
        assert sts[name]['models']['candidate']['wasm_sha256']==r['wasm_sha256']
    audit=json.loads((OUT/'fresh-audit.json').read_text())
    assert not audit['exact'] and not audit['eight_word']
    assert audit['challenge_sha256']==reports['candidate']['inputs']['eval/quality/corpus/persian-domain-hard.json']
    packed=ROOT/'training/pack/out/pilot-fa-en-domain/model.bin'
    assert hashlib.sha256(packed.read_bytes()).hexdigest()==reports['candidate']['model_sha256']
    scores={name:r['fixtures'] for name,r in reports.items()}
    base=scores['round3'];prior=scores['native'];new=scores['candidate']
    lang={name:s['models']['candidate']['languages'] for name,s in sts.items()}
    gates={'validation_eligible':selection['validation']['eligible']}
    for task in ['docs','emoji']:
        gates[task]=new[task]['top1']>=.5 and new[task]['top1']>max(base[task]['top1'],prior[task]['top1'])
    for category in ['negation','numbers']:
        values=[s['logic']['categories'][category]['contrast_accuracy'] for s in [base,prior,new]]
        gates['fresh_'+category]=values[2]>=max(values[:2])
    gates['unseen_corpus']=new['unseen_corpus']['top1']>=max(base['unseen_corpus']['top1'],prior['unseen_corpus']['top1'])
    for language,tolerance in [('en',.002),('fa',.01)]:
        gates['sts_'+language]=lang['candidate'][language]['spearman']>=lang['round3'][language]['spearman']-tolerance
    gates['size']=reports['candidate']['gzip_bytes']<=min(reports[n]['gzip_bytes'] for n in ['round3','native'])*1.005
    for i in [2,3,4]:
        for language in ['en','fa']:
            key=f'round{i}_{language}';gates[key+'_ranking']=new[key]['ndcg@10']>=base[key]['ndcg@10']-.02
    rng=np.random.default_rng(892);intervals={}
    for reference in ['round3','native']:
        for task in ['docs','emoji','unseen_corpus']:
            a=scores[reference][task]['queries'];b=new[task]['queries']
            assert [q['id'] for q in a]==[q['id'] for q in b]
            delta=np.array([float(y['top1_correct'])-float(x['top1_correct']) for x,y in zip(a,b)])
            boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            intervals[reference+'_'+task]=dict(delta=float(delta.mean()),percentile95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(release_ready=all(gates.values()),gates=gates,selection=selection,sts=lang,
                fresh={name:{k:r[k] for k in ['docs','emoji','unseen_corpus','logic']} for name,r in scores.items()},
                sizes={name:{k:r[k] for k in ['wasm_bytes','gzip_bytes']} for name,r in reports.items()},
                packed_bytes=packed.stat().st_size,paired_query_bootstrap=intervals,
                limitations='Small authored fresh set; query bootstrap does not measure author/topic bias. Known-corpus supervision and unseen-corpus evaluation are separate. Historical fixtures are development regressions, not fresh evidence.')
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Technical and colloquial Persian pilot — measured results','',
           'Local run, 2026-10-12. Actual packed WASM, pure cosine; no aliases, reranker or query rules.',
           f"Release ready: **{str(result['release_ready']).lower()}**. Selected `{selection['checkpoint']}` using validation only.",'',
           '| Fresh metric | Round3 | Previous native | Domain candidate |','|---|---:|---:|---:|']
    for task in ['docs','emoji','unseen_corpus','logic']:
        n=new[task]['count']
        lines.append(f'| {task} top1 ({n}) | '+' | '.join(f"{round(scores[m][task]['top1']*n)}/{n}" for m in reports)+' |')
        lines.append(f'| {task} NDCG@10 | '+' | '.join(f"{scores[m][task]['ndcg@10']:.4f}" for m in reports)+' |')
    for cat in ['negation','numbers']:
        lines.append(f'| {cat} strict contrast | '+' | '.join(f"{round(scores[m]['logic']['categories'][cat]['contrast_accuracy']*12)}/12" for m in reports)+' |')
        lines.append(f'| {cat} both directions correct | '+' | '.join(f"{scores[m]['logic']['pairs'][cat]['both_contrasts_correct']}/6" for m in reports)+' |')
    lines+=['','## Retention','', '| Metric | Round3 | Previous native | Candidate |','|---|---:|---:|---:|']
    for language in ['en','fa']:
        lines.append(f'| External {language} STS Spearman | '+' | '.join(f"{lang[m][language]['spearman']:.6f}" for m in reports)+' |')
    for key in ['native_qa','previous_docs','previous_emoji','previous_odyssey']+[f'round{i}_{l}' for i in [2,3,4] for l in ['fa','en']]:
        lines.append(f'| {key} NDCG@10 | '+' | '.join(f"{scores[m][key]['ndcg@10']:.4f}" for m in reports)+' |')
    lines+=['','## Gates and size','']+[f'- {k}: {"PASS" if v else "FAIL"}' for k,v in gates.items()]
    lines+=['',f"Packed weights: {packed.stat().st_size:,} bytes; 9,901,952 parameters; unchanged 32k tokenizer.",
            '', '| Bundle | Round3 | Previous native | Candidate |','|---|---:|---:|---:|']
    for field in ['wasm_bytes','gzip_bytes']:lines.append(f'| {field} | '+' | '.join(f'{reports[m][field]:,}' for m in reports)+' |')
    lines+=['','## Scope and evidence','',result['limitations'],
            '','300 training queries and 100 validation queries supervise 30 known docs and 70 emoji. The 76 new queries were frozen before training; 12 additionally use a new document corpus. The 538,417-text audit found no normalized exact or eight-word overlap for the 112 held-out query/new-document texts.',
            '', 'The original six-epoch run and its three calibration candidates failed validation. A separately recorded retention repair increased Persian replay and historical ranking preservation while lowering learning rates; see the [protocol](persian-pilot-domain.md). All logs and rejected checkpoints are preserved.',
            '', 'Raw per-query ranks, contrast margins, paired-direction results, source hashes and bootstrap intervals are in `dist/pilot-fa-en-domain/`. Preview: `/eval/quality/native-preview.html`.']
    lines+=['','## What improved and what remains broken','',
            '- Fresh docs: queued state updates now retrieve the functional setter document; lifting state, cleanup, refs, context and reducers also improve. One dependency-array query still incorrectly retrieves cleanup.',
            '- Fresh colloquial emoji: the champion paraphrase retrieves 🏆; fever retrieves 🤒; debugging retrieves 🐛. Known user regressions also pass without aliases: محبت → 😍 and «بهترینِ تمام دوران» → 🏆; a literal goat remains 🐐.',
            '- Numeric contrast at 6/12 is not successful reasoning: both-direction accuracy is 0/6. The model returns 32 remaining jobs for both 37−9+4 and 37−9−4, although the second answer is 24. The non-regression gate passing does not imply numeric competence.',
            '- Complex negation declines to 6/12; only 1/6 pairs is correct in both directions. Having a necessary approval is confused with missing approval even when other conditions are explicitly unknown.',
            '- English STS is stable, but historical round2 English ranking drops from .8453 to .8248 (top1 11/18 to 10/18). New-corpus retrieval improves over the native pilot but remains below round3; native QA also loses the previous native pilot gain.',
            '', 'The selected artifact is the original run epoch2 diagnostic checkpoint. All nine training epochs completed (~15.5 minutes of epoch runtime); none of the six validation interpolation candidates passed every gate. No fresh candidate score was used for checkpoint selection.',
            '', 'Validation: 31 Python tests passed; int4 source/packed inference matched exactly for 32 samples; actual WASM parity passed 148 cases (max absolute difference .000298389, minimum cosine .999998351). Browser docs and emoji checks use the same pure-cosine artifact. These checks establish implementation parity, not general language competence.',
            '', 'No model was published or installed as the default example model. Existing source-data licensing and pretraining exposure have not been re-audited in this experiment.']
    (ROOT/'docs/persian-pilot-domain-results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(release_ready=result['release_ready'],gates=gates,sts=lang,sizes=result['sizes']),indent=2))


if __name__=='__main__':main()
