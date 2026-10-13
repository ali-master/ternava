"""Report every fixed-size reasoning gate; plot measured values, never mock results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/pilot-fa-en-reasoning'
NAMES=['round3','domain','candidate']


def summarize():
    reports={n:json.loads((OUT/f'eval-{n}.json').read_text()) for n in NAMES}
    paths=dict(round3='pilot-fa-en-v3',domain='pilot-fa-en-domain',candidate='pilot-fa-en-reasoning')
    sts={n:json.loads((ROOT/f'dist/{paths[n]}/sts-candidate.json').read_text()) for n in NAMES}
    for n,r in reports.items():
        assert r['inputs']==reports['round3']['inputs']
        assert r['tokenizer_sha256']==reports['round3']['tokenizer_sha256']
        assert sts[n]['dataset_sha256']==sts['round3']['dataset_sha256']
        assert sts[n]['models']['candidate']['wasm_sha256']==r['wasm_sha256']
    audit=json.loads((OUT/'fresh-audit.json').read_text());assert not audit['normalized_exact_collisions'] and not audit['shared_eight_word_spans']
    assert audit['challenge_sha256']==reports['candidate']['inputs']['eval/quality/corpus/persian-reasoning-hard.json']
    packed=ROOT/'training/pack/out/pilot-fa-en-reasoning/model.bin';assert hashlib.sha256(packed.read_bytes()).hexdigest()==reports['candidate']['model_sha256']
    selection=json.loads((OUT/'selection.json').read_text());scores={n:r['fixtures'] for n,r in reports.items()};new=scores['candidate'];prior=scores['domain'];base=scores['round3']
    languages={n:s['models']['candidate']['languages'] for n,s in sts.items()}
    gates=dict(validation_eligible=selection['validation']['eligible'])
    for task in ['docs','emoji']:
        gates[task+'_retention']=new[task]['top1']>=prior[task]['top1']
        gates['fresh_'+task]=new['fresh_'+task]['top1']>=prior['fresh_'+task]['top1']
    for lang in ['fa','en']:
        for cat in ['negation','numbers']:
            key=f'fresh_{lang}_{cat}';metric=new[key]
            gates[key+'_nonregression']=metric['categories'][cat]['contrast_accuracy']>=max(scores[n][key]['categories'][cat]['contrast_accuracy'] for n in ['round3','domain'])
            if lang=='fa':gates[key+'_competence']=metric['pairs'][cat]['both_contrasts_correct']>=4 and metric['categories'][cat]['contrast_accuracy']>=10/12
    for i in [2,3,4]:
        gates[f'round{i}_en_recovered']=new[f'round{i}_en']['ndcg@10']>=base[f'round{i}_en']['ndcg@10']-1e-6
        gates[f'round{i}_fa_retained']=new[f'round{i}_fa']['ndcg@10']>=prior[f'round{i}_fa']['ndcg@10']-.01
    for name in ['unseen_corpus','native_qa','previous_docs','previous_emoji','previous_odyssey']:
        gates[name+'_retention']=new[name]['top1']>=prior[name]['top1']
    for lang,tolerance in [('en',.002),('fa',.005)]:gates['sts_'+lang]=languages['candidate'][lang]['spearman']>=languages['domain'][lang]['spearman']-tolerance
    gates['raw_size_fixed']=packed.stat().st_size==5165680
    gates['wasm_size_fixed']=reports['candidate']['wasm_bytes']==reports['domain']['wasm_bytes']
    gates['wire_size']=abs(reports['candidate']['gzip_bytes']/reports['domain']['gzip_bytes']-1)<=.005
    rng=np.random.default_rng(721);bootstrap={}
    for reference in ['round3','domain']:
        for key in ['fresh_docs','fresh_emoji','fresh_fa_negation','fresh_fa_numbers']:
            a=scores[reference][key]['queries'];b=new[key]['queries'];assert [q['id'] for q in a]==[q['id'] for q in b]
            field='contrast_correct' if key.endswith(('negation','numbers')) else 'top1_correct'
            d=np.array([float(y[field])-float(x[field]) for x,y in zip(a,b)])
            if key.endswith(('negation','numbers')):d=d.reshape(-1,2).mean(1)
            values=d[rng.integers(len(d),size=(10000,len(d)))].mean(1)
            bootstrap[reference+'_'+key]=dict(delta=float(d.mean()),percentile95=np.quantile(values,[.025,.975]).tolist(),unit='pair' if len(d)==6 else 'query')
    result=dict(release_ready=all(gates.values()),gates=gates,selection=selection,sts=languages,
                sizes={n:{k:r[k] for k in ['wasm_bytes','gzip_bytes']} for n,r in reports.items()},packed_bytes=packed.stat().st_size,
                fresh={n:{k:v for k,v in s.items() if k.startswith('fresh_')} for n,s in scores.items()},bootstrap=bootstrap)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Paired reasoning pilot — measured results','',
           'Actual packed WASM on 2026-10-12. Pure cosine, no aliases, calculator, query rewrite or reranker.',
           f"Release ready: **{str(result['release_ready']).lower()}**. Selected epoch {selection['epoch']} using validation only.",'',
           '| Fresh metric | Round 3 | Domain baseline | Reasoning candidate |','|---|---:|---:|---:|']
    for task in ['fresh_docs','fresh_emoji','fresh_fa_negation','fresh_fa_numbers','fresh_en_negation','fresh_en_numbers']:
        n=new[task]['count'];lines.append(f'| {task} Top-1 | '+' | '.join(f"{round(scores[m][task]['top1']*n)}/{n}" for m in NAMES)+' |')
        lines.append(f'| {task} NDCG@10 | '+' | '.join(f"{scores[m][task]['ndcg@10']:.4f}" for m in NAMES)+' |')
        if task.endswith(('negation','numbers')):
            cat=task.split('_')[-1]
            lines.append(f'| {task} strict contrast | '+' | '.join(f"{round(scores[m][task]['categories'][cat]['contrast_accuracy']*n)}/{n}" for m in NAMES)+' |')
            lines.append(f'| {task} both directions | '+' | '.join(f"{scores[m][task]['pairs'][cat]['both_contrasts_correct']}/6" for m in NAMES)+' |')
    lines+=['','## Retention and size','', '| Metric | Round 3 | Domain baseline | Candidate |','|---|---:|---:|---:|']
    for lang in ['en','fa']:lines.append(f'| External {lang} STS Spearman | '+' | '.join(f"{languages[n][lang]['spearman']:.6f}" for n in NAMES)+' |')
    for key in ['docs','emoji','unseen_corpus','logic','native_qa','previous_docs','previous_emoji','previous_odyssey']+[f'round{i}_{l}' for i in [2,3,4] for l in ['fa','en']]:
        lines.append(f'| {key} NDCG@10 | '+' | '.join(f"{scores[m][key]['ndcg@10']:.4f}" for m in NAMES)+' |')
    for key in ['wasm_bytes','gzip_bytes']:lines.append(f'| {key} | '+' | '.join(f"{reports[m][key]:,}" for m in NAMES)+' |')
    lines+=['',f"Packed weights: **{packed.stat().st_size:,} bytes**; 9,901,952 parameters; unchanged 32k tokenizer.",'','## Gates','']
    lines += [f'- {k}: {"PASS" if v else "FAIL"}' for k,v in gates.items()]
    metrics=[json.loads(x) for x in (ROOT/'training/distill/runs/mini-fa-en-reasoning/metrics.jsonl').read_text().splitlines()]
    lines += ['','## Evidence and limits','',f"All six epochs completed: {sum(m['seconds'] for m in metrics):.1f} seconds of epoch runtime. The [fixed protocol](persian-pilot-reasoning.md) records the recipe, source checkpoints and selection rules.",
              f"The overlap audit checked {audit['train_val_sentences']:,} text instances against {audit['texts_checked']} fresh query/document texts: no exact or eight-word matches. Teacher pretraining exposure is unknown.",
              'The 64 authored questions are small and correlated. Docs/emoji use known corpora; their questions are new. English logic includes related translations. Numeric success on boundary/direction questions is not evidence of arithmetic; see pair-level results below.',
              'Historical domain and earlier round challenges are now development/regression data. Bootstrap intervals are descriptive and do not cover author/topic bias. Raw per-query ranks, strict margins, hashes and confidence intervals are in `dist/pilot-fa-en-reasoning/`.',
              '', '## Fresh Persian pair details','', '| Category/pair | Domain correct contrasts | Candidate correct contrasts | Candidate full-corpus Top-1 |','|---|---:|---:|---:|']
    for cat in ['negation','numbers']:
        a=prior['fresh_fa_'+cat]['queries'];b=new['fresh_fa_'+cat]['queries']
        for i in range(6):
            aa=a[2*i:2*i+2];bb=b[2*i:2*i+2]
            lines.append(f'| {cat}/{i+1} | {sum(q["contrast_correct"] for q in aa)}/2 | {sum(q["contrast_correct"] for q in bb)}/2 | {sum(q["top1_correct"] for q in bb)}/2 |')
    lines += ['', '## Interpretation and implementation checks', '',
              'The selected artifact is the English-safe transplant of epoch six with alpha 1: all newly trained Persian-script rows are retained, while shared and non-Persian weights return to round three. All six raw epochs and all three calibration candidates fail at least one eligibility gate. Nothing is promoted to the default examples.',
              'Fresh docs and emoji Top-1 are unchanged from the domain baseline (3/8 and 4/8). Previous-domain regression Top-1 improves from 15/20 to 16/20 and 12/20 to 13/20; these reused cases are not new generalization evidence.',
              'Fresh Persian negation deteriorates from 9/12 to 6/12 strict contrasts and from 4/6 to 2/6 correct pairs. Full-corpus Top-1 falls from 7/12 to 4/12. The model reverses both sides of the ban-revocation and neither/nor counterfactuals. This is a failed reasoning candidate.',
              'Numeric strict contrasts rise from 6/12 to 7/12. The percentage pair is the one newly correct contrast pair, but the discount query retrieves a shelf-count answer ahead of the correct fare. Both-directions full-corpus Top-1 remains 0/6. Stock, fractions, elapsed time and strict bounds retain fixed-answer shortcuts.',
              'The development diagnostic separates memorization from transfer: on 32 seen two-operation Persian pairs, both-direction successes rise from 3 to 18; on 20 validation pairs with different wording and ranges, both checkpoints score 0. This supports a generalization failure; it does not prove a hard architectural impossibility. The same split changes wording and numbers together, so their separate effects are not identified.',
              'Actual WASM English embeddings are bit-identical to round three for 2,533 external sentences (maximum absolute difference 0). The three historical English ranking scores also exactly match round three. Domain-era gains on English rounds three and four are rolled back, rather than preserved; all scores are listed above.',
              '37 Python tests passed. Source versus packed int4 inference matched exactly on 32 samples. Actual WASM parity passed 136 cases: maximum absolute difference 0.000406154, minimum cosine 0.999996419. Browser verification found no console errors. These are implementation checks, not evidence of semantic correctness.',
              'No architecture or tokenizer change; packed weights and WASM are byte-count identical to the prior domain model. Gzip grows by 227 bytes (about 0.0042%). Optional wasm-opt optimization was unavailable, consistently with the compared builds.',
              '', 'README screenshots show a real, known successful Persian query, not a cherry-picked fresh success presented as aggregate evidence. The quality chart is generated directly from the saved evaluation reports. No weights were published.']
    (ROOT/'docs/persian-pilot-reasoning-results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(release_ready=result['release_ready'],gates=gates,sizes=result['sizes']),indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((OUT/'summary.json').read_text());colors=['#8d9aa9','#65b5e8','#f0bb65']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#e5eaf3','axes.labelcolor':'#c4cede','xtick.color':'#c4cede','ytick.color':'#c4cede'})
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),facecolor='#111821');labels=['Round 3','Domain','Reasoning']
    for ax in axes:ax.set_facecolor('#111821');ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#455166');ax.grid(axis='y',color='#455166',alpha=.35);ax.set_axisbelow(True)
    groups=[['fresh_docs','fresh_emoji'],['fresh_fa_negation','fresh_fa_numbers']]
    for a,keys in enumerate(groups):
        for j,n in enumerate(NAMES):
            values=[]
            for key in keys:
                f=data['fresh'][n][key]
                values.append(f['top1']*100 if a==0 else f['pairs'][key.split('_')[-1]]['both_contrasts_correct']/6*100)
            bars=axes[a].bar(np.arange(2)+(j-1)*.24,values,.22,color=colors[j],label=labels[j])
            axes[a].bar_label(bars,labels=[f'{round(v*(8 if a==0 else 6)/100)}/{8 if a==0 else 6}' for v in values],padding=5,color=colors[j],fontsize=11)
        axes[a].set_ylim(0,112);axes[a].set_yticks([0,25,50,75,100],['0%','25%','50%','75%','100%']);axes[a].set_xticks([0,1],['Technical docs','Colloquial emoji'] if a==0 else ['Complex negation','Numeric reasoning'])
        axes[a].set_title('Fresh Persian retrieval · Top-1' if a==0 else 'Both opposite queries correct · strict margin',color='#e5eaf3',pad=18,fontsize=13)
    axes[0].legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.14),labelcolor='#d4ddea')
    fig.suptitle('Persian mini: measured ability, same model size',x=.07,ha='left',fontsize=21,fontweight='bold')
    fig.text(.07,.89,'64 frozen FA/EN questions · actual WASM · pure cosine · no aliases',color='#9baec5')
    fig.text(.07,.04,f"Packed weights: {data['packed_bytes']:,} bytes   |   WASM: {data['sizes']['candidate']['wasm_bytes']:,} bytes   |   Experimental; release gates {'PASS' if data['release_ready'] else 'FAIL'}",fontsize=10,color='#9baec5')
    fig.subplots_adjust(left=.07,right=.97,top=.77,bottom=.23,wspace=.25)
    path=ROOT/'docs/assets/persian-quality.png';fig.savefig(path,dpi=160,facecolor=fig.get_facecolor());print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plot-only',action='store_true');args=p.parse_args()
    plot() if args.plot_only else summarize()
