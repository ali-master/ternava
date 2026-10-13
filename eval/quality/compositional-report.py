"""Report the frozen factorial diagnosis, controlled training and actual WASM outcome."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dist/pilot-fa-en-compositional'
NAMES=['current','control','candidate']


def summarize():
    reports={n:json.loads((OUT/f'eval-{n}.json').read_text()) for n in NAMES}
    scores={n:r['fixtures'] for n,r in reports.items()};a=scores['current'];b=scores['candidate']
    selections={n:json.loads((ROOT/f'dist/pilot-fa-en-compositional{"-control" if n=="control" else ""}/selection.json').read_text()) for n in NAMES[1:]}
    for name,s in selections.items():assert hashlib.sha256((ROOT/s['checkpoint']).read_bytes()).hexdigest()==s['sha256']
    assert all(r['inputs']==reports['current']['inputs'] and r['tokenizer_sha256']==reports['current']['tokenizer_sha256'] for r in reports.values())
    paths={'current':ROOT/'dist/pilot-fa-en-reasoning/sts-candidate.json',**{n:OUT/f'sts-{n}.json' for n in NAMES[1:]}}
    sts={};dataset=None
    for n,path in paths.items():
        r=json.loads(path.read_text());m=r['models']['candidate' if n=='current' else n]
        assert m['wasm_sha256']==reports[n]['wasm_sha256']
        if dataset is not None:assert r['dataset_sha256']==dataset
        dataset=r['dataset_sha256'];sts[n]=m['languages']
    audit=json.loads((OUT/'fresh-audit.json').read_text());assert not audit['normalized_exact_collisions']
    assert audit['challenge_sha256']==reports['candidate']['inputs']['eval/quality/corpus/persian-compositional-hard.json']
    for s in selections.values():assert s['challenge_sha256']==audit['challenge_sha256']
    english=json.loads((OUT/'english-retention.json').read_text())
    known=json.loads((OUT/'known-queries.json').read_text())
    assert all(known[n]['wasm_sha256']==reports[n]['wasm_sha256'] for n in NAMES)
    packed=ROOT/'training/pack/out/pilot-fa-en-compositional/model.bin'
    assert hashlib.sha256(packed.read_bytes()).hexdigest()==reports['candidate']['model_sha256']
    gates=dict(validation=all(selections['candidate']['validation']['gates'].values()),english_exact=english['candidate']['vectors_bit_exact'],packed_size=packed.stat().st_size==5165680,wasm_size=reports['candidate']['wasm_bytes']==reports['current']['wasm_bytes'],gzip_size=abs(reports['candidate']['gzip_bytes']/reports['current']['gzip_bytes']-1)<=.005)
    gates['known_user_queries']=all(q['pass_case'] for q in known['candidate']['queries'])
    for task in ['docs','emoji','previous_docs','previous_emoji','reasoning_docs','reasoning_emoji','fresh_docs','fresh_emoji']:
        gates[task]=all(b[task][k]>=a[task][k]-1e-6 for k in ['top1','ndcg@10'])
    for i in [2,3,4]:
        gates[f'round{i}_en']=abs(b[f'round{i}_en']['ndcg@10']-a[f'round{i}_en']['ndcg@10'])<1e-6
        gates[f'round{i}_fa']=b[f'round{i}_fa']['ndcg@10']>=a[f'round{i}_fa']['ndcg@10']-.01
    for task in ['native_qa','previous_odyssey','unseen_corpus','logic','reasoning_fa_negation','reasoning_fa_numbers']:
        gates[task]=b[task]['top1']>=a[task]['top1']-1e-6
    gates['sts_fa']=sts['candidate']['fa']['spearman']>=sts['current']['fa']['spearman']-.002
    for lang in ['fa','en']:
        for cat in ['negation','numbers']:
            key=f'fresh_{lang}_{cat}';new=b[key];prior=a[key]
            gates[key+'_retention']=new['categories'][cat]['contrast_accuracy']>=prior['categories'][cat]['contrast_accuracy'] and new['top1']>=prior['top1']
            if lang=='fa':gates[key+'_functional']=new['categories'][cat]['contrast_accuracy']>=14/16 and new['pairs'][cat]['both_top1_correct']>=6
    result=dict(release_ready=all(gates.values()),gates=gates,selection=selections,sts=sts,english=english,
                packed_bytes=packed.stat().st_size,sizes={n:{k:r[k] for k in ['wasm_bytes','gzip_bytes']} for n,r in reports.items()},
                fresh={n:{k:v for k,v in r.items() if k.startswith('fresh_')} for n,r in scores.items()})
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Compositional pilot: diagnosis and measured outcome','','Actual WASM evaluation on 2026-10-13. Pure cosine on full corpora; no aliases, calculator, language routing, query rewriting or reranker.',
           f'**Release ready: {str(result["release_ready"]).lower()}.** Existing examples and the comparison page remain unchanged; failed candidates are diagnostic only.',
           '', '## Root cause before training','',
           'See the [factorial diagnosis and frozen protocol](persian-pilot-compositional.md). Familiar numeric pair accuracy exaggerated arithmetic ability: current both-direction wins were 47/96, falling to 14/96 with adjacent answers and 12/96 with only the wording changed. Negation fell from 144/144 familiar probes to 57/144 with new wording, while noun substitution stayed 144/144. These repeated template probes are not independent natural-language observations.',
           'No unknown tokens or truncation occurred. fp32 embeddings did not restore generalization. English-safe weight restoration removed some numeric gains, but the raw checkpoint already failed new wording. The evidence identifies lexical/template shortcuts and an underconstrained two-choice objective; it does not prove that general arithmetic is impossible for this architecture.',
           '', '## Final comparison','', '| Fresh metric | Current | Binary control | Revised objective |','|---|---:|---:|---:|']
    for task in ['docs','emoji','fa_negation','fa_numbers','en_negation','en_numbers']:
        key='fresh_'+task;count=b[key]['count']
        lines.append(f'| {task} full-corpus Top-1 | '+' | '.join(f'{round(scores[n][key]["top1"]*count)}/{count}' for n in NAMES)+' |')
        lines.append(f'| {task} NDCG@10 | '+' | '.join(f'{scores[n][key]["ndcg@10"]:.4f}' for n in NAMES)+' |')
        if task.startswith(('fa_','en_')):
            cat=task.split('_')[-1]
            lines.append(f'| {task} strict contrast | '+' | '.join(f'{round(scores[n][key]["categories"][cat]["contrast_accuracy"]*count)}/{count}' for n in NAMES)+' |')
            for field in ['both_contrasts_correct','both_top1_correct']:
                lines.append(f'| {task} {field} | '+' | '.join(f'{scores[n][key]["pairs"][cat][field]}/{count//2}' for n in NAMES)+' |')
    lines+=['','## Preservation','', '| Metric | Current | Binary control | Revised objective |','|---|---:|---:|---:|']
    for lang in ['en','fa']:lines.append(f'| External {lang} STS | '+' | '.join(f'{sts[n][lang]["spearman"]:.6f}' for n in NAMES)+' |')
    for key in ['docs','emoji','native_qa','unseen_corpus','previous_docs','previous_emoji','previous_odyssey','reasoning_docs','reasoning_emoji','reasoning_fa_negation','reasoning_fa_numbers']+[f'round{i}_{l}' for i in [2,3,4] for l in ['fa','en']]:
        lines.append(f'| {key} NDCG@10 | '+' | '.join(f'{scores[n][key]["ndcg@10"]:.4f}' for n in NAMES)+' |')
    for k in ['wasm_bytes','gzip_bytes']:lines.append(f'| {k} | '+' | '.join(f'{reports[n][k]:,}' for n in NAMES)+' |')
    lines+=['',f'Packed weights: **{packed.stat().st_size:,} bytes**; 9,901,952 parameters and unchanged 32k tokenizer.']
    for name,r in english.items():lines.append(f'{name}: {r["texts"]:,} English embeddings bit-exact to current: **{r["vectors_bit_exact"]}**; maximum absolute difference {r["maximum_absolute_difference"]}.')
    lines+=['','Known user-query regression results (not fresh evidence):']
    for q in known['candidate']['queries']:lines.append(f'- `{q["query"]}` → {", ".join(q["top3"])}; pass: {q["pass_case"]}.')
    lines+=['','## Validation and objective ablation','']
    for n in NAMES[1:]:
        run=ROOT/f'training/distill/runs/mini-fa-en-compositional{"-control" if n=="control" else ""}'
        history=json.loads((run/'metrics.json').read_text());base=json.loads((run/'baseline.json').read_text());s=selections[n]['validation']
        lines.append(f'{n}: selected epoch {s["epoch"]}; {len(history)} × 250 steps, {sum(r["seconds"] for r in history):.1f} seconds of epoch runtime. Passing validation gates: {sum(s["gates"].values())}/{len(s["gates"])}.')
        for cat in ['negation','numbers']:
            v=s['logic'][cat];old=base['logic'][cat]
            lines.append(f'- {cat}: both-view exact {old["both_views"]:.3f} → {v["both_views"]:.3f}; full validation-corpus Top-1 {old["full_top1"]:.3f} → {v["full_top1"]:.3f}.')
    after=json.loads((OUT/'diagnosis-after.json').read_text())
    lines+=['','Post-training replay of the same controlled development probes:', '', '| Probe | Binary control | Revised objective |','|---|---:|---:|']
    for category,axis,field in [('numeric','seen','exact_both'),('numeric','wording','both'),('negation','wording','both')]:
        values=[]
        for n in NAMES[1:]:
            rows=[v for k,v in after[n]['buckets'].items() if k.startswith(category) and k.endswith('/'+axis)]
            values.append(f'{sum(r[field] for r in rows)}/{sum(r["pairs"] for r in rows)}')
        lines.append(f'| {category}, {axis}, {field} | '+' | '.join(values)+' |')
    lines+=['','Both arms use the same corrected data, initialization, frozen weights, seed, replay and step budget. The ablation changes the combined objective: full-batch negatives, adjacent-answer supervision and equivalent-query consistency. It does not separately identify the contribution of each term. Checkpoints are selected independently on validation, never on final scores.',
            '', '## Gates','']+[f'- {k}: {"PASS" if v else "FAIL"}' for k,v in gates.items()]
    lines+=['','## Fresh Persian failures','', '| Case | Current contrast / full Top-1 | Revised contrast / full Top-1 | Revised top answer |','|---|---:|---:|---|']
    for cat in ['negation','numbers']:
        key='fresh_fa_'+cat
        assert [q['id'] for q in a[key]['queries']]==[q['id'] for q in b[key]['queries']]
        for x,y in zip(a[key]['queries'],b[key]['queries']):
            if not y['top1_correct'] or not y['contrast_correct']:
                lines.append(f'| {y["id"]} | {int(x["contrast_correct"])}/{int(x["top1_correct"])} | {int(y["contrast_correct"])}/{int(y["top1_correct"])} | {y["top_text"].replace("|","/")} |')
    lines+=['','## Limits and provenance','',
            f'The final audit checked {audit["texts_checked"]} query/new-document texts against {audit["train_val_sentences"]:,} training/development text instances: no normalized full-text collision; {len(audit["shared_eight_word_spans"])} shared eight-word spans, disclosed below. Known docs/emoji corpus documents are intentionally supervised. Teacher pretraining exposure is unknown. The 64 authored questions are small, related, and not a representative Persian benchmark.',
            *[f'- Shared span: `{s}`' for s in audit['shared_eight_word_spans']],
            '', 'The preliminary invalid-validation run was interrupted and excluded; it is preserved separately. Tests now reject identical validation views and a one-class boundary set. No fresh-model scores were used to repair that data.',
            'The final fixture is now consumed regression evidence. Future model selection needs a new final test. The frozen-backbone design guarantees retention only for texts using unchanged embedding rows; mixed-language English containing changed Persian tokens may differ.',
            'Source/data/checkpoint hashes, all per-query ranks, model sizes, logs and English comparisons are retained in `dist/pilot-fa-en-compositional/`. Training and evaluation source is versioned; generated checkpoint and WASM files remain local. Default examples and the existing comparison page are unchanged. No model was published.']
    lines+=['','## Interpretation and checks','',
            'The revised objective beats the matched binary control on this final retrieval set, but does not establish logical generalization. It retrieves 8/8 new docs versus 7/8 for current/control and keeps new emoji at 5/8 versus 5/8 current and 4/8 control. Persian negation remains 0/8 complete counterfactual pairs; numeric full-corpus pairs remain 1/8, equal to current. Historical docs/emoji and external Persian STS also have failed preservation gates. Do not promote this checkpoint.',
            'This is not solely a Persian coverage problem: the unchanged English baseline solves zero complete pairs in both new English negation and numeric sets (0/4 each). Preserving English embeddings preserves their existing reasoning weaknesses as well as their retrieval quality. The data/objective fixes remove exploitable supervision patterns, but embedding-row-only adaptation has not taught the missing computation. These runs do not establish an impossibility result for the fixed-size architecture.',
            'A subsequent experiment should target the shared reasoning representation with bilingual compositional supervision while constraining English/domain ranking, rather than another embedding-only run or a post-training weight transplant. Keep the same parameter budget and a new final test; any candidate still has to pass the existing retention gates. This is the next hypothesis, not an already demonstrated solution.',
            '44 Python tests pass, including truth-table labels, exact arithmetic labels, nearest-answer negatives, semantic duplicate masking, distinct validation views and balanced boundary labels. Source/packed int4 parity is exact on 32 samples for each selected arm. Candidate WASM parity passes 136 tokenization/vector cases (maximum absolute error 0.000349149, minimum cosine 0.999998072). Optional wasm-opt was unavailable, as with the baseline. The UI was not changed or re-tested.',
            '', '## Reproduction in the existing local pilot workspace','',
            'The commands intentionally refuse to overwrite prepared data or runs. The earlier checkpoints and caches must already exist, as documented in preceding pilot reports. Results and final tests from this run are now consumed.',
            '```bash', '.venv/bin/python training/distill/diagnose_reasoning.py',
            'PYTHONPATH=training/distill .venv/bin/python -m prep.compositional',
            '.venv/bin/python training/distill/compositional.py',
            '.venv/bin/python training/distill/compositional.py --control',
            '# Pack/build each selected.pt with the existing int4 pack and build-engine commands.',
            '.venv/bin/python eval/quality/domain.py --model candidate=dist/pilot-fa-en-compositional/d256/tern_engine.js --output dist/pilot-fa-en-compositional --challenge eval/quality/corpus/persian-compositional-hard.json',
            '```']
    (ROOT/'docs/persian-pilot-compositional-results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(release_ready=result['release_ready'],gates=gates,sizes=result['sizes']),indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data=json.loads((OUT/'summary.json').read_text());colors=['#8597ad','#67b5de','#e9ba6e']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':'#e8edf5','xtick.color':'#bcc8da','ytick.color':'#bcc8da'})
    fig,axes=plt.subplots(1,2,figsize=(13,5.6),facecolor='#111821')
    for ax in axes:
        ax.set_facecolor('#111821');ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#455166');ax.grid(axis='y',color='#455166',alpha=.35);ax.set_axisbelow(True)
        ax.set_ylim(0,110);ax.set_yticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
    for j,n in enumerate(NAMES):
        for i,keys in enumerate([['docs','emoji'],['fa_negation','fa_numbers']]):
            values=[]
            for key in keys:
                r=data['fresh'][n]['fresh_'+key]
                values.append(round(r['top1']*r['count']) if i==0 else r['pairs'][key.split('_')[-1]]['both_top1_correct'])
            bars=axes[i].bar(np.arange(2)+(j-1)*.24,np.array(values)/8*100,.22,color=colors[j],label=['Current','Binary control','Revised objective'][j])
            axes[i].bar_label(bars,labels=[f'{v}/8' for v in values],padding=5,color=colors[j])
    axes[0].set_xticks([0,1],['Technical docs','Colloquial emoji']);axes[1].set_xticks([0,1],['Negation','Numbers'])
    axes[0].set_title('Fresh retrieval: full-corpus Top-1',color='#e8edf5',pad=16)
    axes[1].set_title('Both queries retrieve the correct answer first',color='#e8edf5',pad=16)
    axes[0].legend(frameon=False,ncol=3,loc='upper left',bbox_to_anchor=(-.05,-.14),labelcolor='#bcc8da',fontsize=10)
    fig.suptitle('Diagnose the shortcut, then test the training change',x=.06,ha='left',fontsize=20,fontweight='bold')
    fig.text(.06,.89,'64 new FA/EN queries · actual WASM · no aliases · fixed architecture',color='#9caec5')
    fig.text(.06,.04,f'Packed weights: {data["packed_bytes"]:,} bytes | English vector equality checked | Release gates: {"PASS" if data["release_ready"] else "FAIL"}',fontsize=10,color='#9caec5')
    fig.subplots_adjust(left=.06,right=.98,top=.76,bottom=.23,wspace=.25)
    path=ROOT/'docs/assets/persian-compositional-quality.png';fig.savefig(path,dpi=160,facecolor=fig.get_facecolor());print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plot-only',action='store_true');a=p.parse_args();plot() if a.plot_only else summarize()
