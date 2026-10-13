"""Reproducible supervised round-two data; never train on challenge/test labels."""
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'training/distill'))
from tokenization import bilingual_normalizer, load_tokenizer
from prep.ingest import teacher_encode

OUT = ROOT / 'training/distill/cache/mini-fa-en-round2'
OLD = ROOT / 'training/distill/cache/mini-fa-en-32k'
normalize_text = bilingual_normalizer().normalize_str

def normalize(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', normalize_text(text)).split())

def number(n, lang):
    units = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
             if lang == 'en' else 'صفر یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده هجده نوزده'.split())
    tens = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety'] if lang == 'en' else ['','','بیست','سی','چهل','پنجاه','شصت','هفتاد','هشتاد','نود']
    if n < 20:return units[n]
    if n >= 100:return str(n)
    return tens[n//10] + ((' ' if lang == 'en' else ' و ') + units[n%10] if n%10 else '')

def synthetic(split, count=6000):
    """Different entities/ranges across splits; symmetric polarity prevents label shortcuts."""
    rng=random.Random(71 if split=='train' else 72); result=[];seen=set()
    for lang in ['fa','en']:
        names = (['علی','رضا','مینا','زهرا','حامد','پروین','شیرین','امید','کامران','بهروز','سعید','نسرین'] if lang=='fa' else ['Alice','Bob','David','Emma','James','Helen','Jack','Laura','Paul','Grace','Henry','Lucy']) if split=='train' else (['پریسا','کیان','الهام','بهنام'] if lang=='fa' else ['Arthur','Diana','George','Irene'])
        verbs = [('دید','دیده شد','saw','seen'),('دنبال کرد','دنبال شد','followed','followed'),('تحسین کرد','تحسین شد','praised','praised'),('دعوت کرد','دعوت شد','invited','invited'),('نجات داد','نجات داده شد','rescued','rescued'),('معرفی کرد','معرفی شد','introduced','introduced')]
        for i in range(count):
            a,b=rng.sample(names,2);t=rng.randrange(3);cat=['role-order','negation','numbers'][i%3]
            if cat=='role-order':
                fa,passive,en,enpass=rng.choice(verbs)
                if lang=='fa':
                    p=f'{a} {b} را {fa}.';n=f'{b} {a} را {fa}.'
                    q=[f'{b} توسط {a} {passive}.',f'کسی که {b} را {fa}، {a} بود.',f'{a} بود که {b} را {fa}.'][t]
                else:
                    p=f'{a} {en} {b}.';n=f'{b} {en} {a}.'
                    q=[f'{b} was {enpass} by {a}.',f'The person who {en} {b} was {a}.',f'It was {a} who {en} {b}.'][t]
            elif cat=='negation':
                obj=rng.choice(['سفر','خرید','ثبت نام','ارسال نامه','دیدار','آموزش','ورزش','پرداخت','مطالعه','تماس'] if lang=='fa' else ['travel','shopping','registration','sending a letter','a visit','training','exercise','payment','study','a phone call'])
                positive=rng.choice([True,False])
                if lang=='fa':
                    yes=f'{a} {obj} را انجام داد.';no=f'{a} {obj} را انجام نداد.'
                    q=([f'{a} از انجام {obj} صرف نظر نکرد و آن را انجام داد.',f'این درست نیست که {a} {obj} را انجام نداده باشد.',f'{a} موفق به انجام {obj} شد.'] if positive else [f'{a} از انجام {obj} صرف نظر کرد.',f'این درست نیست که {a} {obj} را انجام داده باشد.',f'{a} موفق به انجام {obj} نشد.'])[t]
                else:
                    yes=f'{a} completed {obj}.';no=f'{a} did not complete {obj}.'
                    q=([f'{a} did not give up on {obj} and completed it.',f'It is false that {a} did not complete {obj}.',f'{a} succeeded in completing {obj}.'] if positive else [f'{a} gave up on {obj}.',f'It is false that {a} completed {obj}.',f'{a} failed to complete {obj}.'])[t]
                p,n=(yes,no) if positive else (no,yes)
            else:
                x,y=rng.sample(range(2,90) if split=='train' else range(90,120),2)
                xn,yn=number(x,lang),number(y,lang);up=y>x
                if t==0:
                    if lang=='fa':q=f'مقدار از {xn} به {yn} رسید.';p,n=(('مقدار افزایش یافته است.','مقدار کاهش یافته است.') if up else ('مقدار کاهش یافته است.','مقدار افزایش یافته است.'))
                    else:q=f'The value moved from {xn} to {yn}.';p,n=(('The value increased.','The value decreased.') if up else ('The value decreased.','The value increased.'))
                elif t==1:
                    if lang=='fa':q=f'{a} {xn} سکه و {b} {yn} سکه دارد.';p=f'{b if up else a} سکه بیشتری دارد.';n=f'{a if up else b} سکه بیشتری دارد.'
                    else:q=f'{a} has {xn} coins and {b} has {yn} coins.';p=f'{b if up else a} has more coins.';n=f'{a if up else b} has more coins.'
                else:
                    total=max(x,y);used=min(x,y);left=total-used
                    if lang=='fa':q=f'از {number(total,lang)} بلیت، {number(used,lang)} بلیت فروخته شد.';p=f'{number(left,lang)} بلیت باقی مانده است.';n=f'{number(total+used,lang)} بلیت باقی مانده است.'
                    else:q=f'Out of {number(total,lang)} tickets, {number(used,lang)} were sold.';p=f'{number(left,lang)} tickets remain.';n=f'{number(total+used,lang)} tickets remain.'
            key=(q,p,n)
            if key not in seen:result.append({'anchor':q,'positive':p,'negative':n,'lang':lang,'category':cat});seen.add(key)
    return result

def encode_rows(rows, keys, tokenizer):
    encoded=tokenizer([r[k] for r in rows for k in keys],padding='max_length',truncation=True,max_length=128,return_tensors='pt')
    return {k:v.reshape(len(rows),len(keys),128) for k,v in encoded.items() if k in ('input_ids','attention_mask')}

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data.pt').exists():raise FileExistsError('Preserve prepared round-two data')
    old_manifest=json.loads((OLD/'bilingual.manifest.json').read_text());tk=load_tokenizer(old_manifest['tokenizer'])
    held=[]
    for path in [ROOT/'eval/quality/corpus/bilingual-hard-round2.json',ROOT/'eval/quality/corpus/bilingual-hard-heldout.json']:
        fixture=json.loads(path.read_text());held += [r['text'] for r in fixture['queries']+fixture['corpus']]
    external=json.loads((ROOT/'dist/pilot-fa-en/external-sts.json').read_text())
    held += [r[k] for b in external['benchmarks'].values() for r in b['samples'] for k in ['sentence1','sentence2']]
    pairs=json.loads((OLD/'pairs.json').read_text());held += [r[l] for split in ['val','test'] for r in pairs[split] for l in ['en','fa']]
    forbidden={normalize(t) for t in held}
    nli_rev='d482672c8e74ce18da116f430137434ba2e52fab'
    nli=load_dataset('sentence-transformers/all-nli','triplet',revision=nli_rev)
    triplets={};sts={};excluded={};raw={};data={};train_seen=set()
    # Validation first, so no endpoint from it can enter supervised training.
    for split in ['val','train']:
        source=nli['dev' if split=='val' else 'train'].shuffle(seed=73)
        limit=500 if split=='val' else 12000
        candidates=[dict(r,lang='en',category='nli') for r in source.select(range(min(len(source),limit*2)))]
        accepted=[]
        for row in candidates:
            if any(normalize(row[k]) in forbidden for k in ['anchor','positive','negative']):continue
            accepted.append(row)
            if len(accepted)==limit:break
        accepted += [r for r in synthetic(split,600 if split=='val' else 6000) if not any(normalize(r[k]) in forbidden for k in ['anchor','positive','negative'])]
        assert accepted
        if split=='val':forbidden.update(normalize(r['anchor']) for r in accepted)
        triplets[split]=accepted
    for split in ['val','train']:
        rows=[]
        for lang,ds,rev in [('en','mteb/stsbenchmark-sts','96943a16ea6a35129e253c659081cb59daf81b30'),('fa','MCINext/synthetic-persian-sts','e078618092074727e71f57f6bef76656d85b3d1d')]:
            all_splits=load_dataset(ds,revision=rev)
            key='train' if split=='train' else ('validation' if 'validation' in all_splits else 'dev')
            original=all_splits[key] if lang=='en' else all_splits['train'].shuffle(seed=76).select(range(1000) if split=='val' else range(1000,15000))
            selected=[dict(r,lang=lang) for r in original if not any(normalize(r[k]) in forbidden for k in ['sentence1','sentence2'])]
            excluded[f'{split}-{lang}']=len(original)-len(selected)
            rows+=selected
        assert rows
        if split=='val':forbidden.update(normalize(r[k]) for r in rows for k in ['sentence1','sentence2'])
        sts[split]=rows
    # Refilter training after both validation sources were reserved.
    triplets['train']=[r for r in triplets['train'] if not any(normalize(r[k]) in forbidden for k in ['anchor','positive','negative'])]
    for split in ['train','val']:
        data['triplet_'+split]=encode_rows(triplets[split],['anchor','positive','negative'],tk)
        data['sts_'+split]=encode_rows(sts[split],['sentence1','sentence2'],tk)
        data['sts_'+split]['score']=torch.tensor([float(r['score']) for r in sts[split]])
        data['sts_'+split]['lang']=torch.tensor([int(r['lang']=='fa') for r in sts[split]])
    # Replay a balanced, deterministic subset of the original training only.
    original=torch.load(OLD/'bilingual.train.pt',weights_only=False)
    rng=random.Random(74);rng.shuffle(original)
    replay=[];counts={'en':0,'fa':0}
    for row in original:
        if counts[row['source']]<30000 and normalize(row['text']) not in forbidden:
            replay.append(row);counts[row['source']]+=1
    assert min(counts.values())==30000
    extra_texts=list(dict.fromkeys(r[k] for r in triplets['train'] if r['category']=='nli' for k in ['anchor','positive']))[:20000]
    targets=teacher_encode(extra_texts,old_manifest['teacher']['id'],old_manifest['teacher']['revision'],128)
    encoded=tk(extra_texts,padding='max_length',truncation=True,max_length=128,return_tensors='pt')
    data['replay']={k:torch.cat([torch.stack([r[k] for r in replay]),encoded[k] if k!='teacher_emb' else targets]) for k in ['input_ids','attention_mask','teacher_emb']}
    raw={'triplets':triplets,'sts':sts,'replay_texts':[r['text'] for r in replay]+extra_texts}
    (OUT/'texts.json').write_text(json.dumps(raw,ensure_ascii=False)+'\n')
    manifest={**old_manifest,'round':2,'nli_revision':nli_rev,'persian_sts_revision':'e078618092074727e71f57f6bef76656d85b3d1d','validation_note':'Synthetic triplets use different entities/numeric ranges; generic positive/negative sentences may recur. Persian STS validation endpoints excluded from training.','supervised_counts':{k:len(v['input_ids']) for k,v in data.items()},'excluded_overlap':excluded,'texts_sha256':hashlib.sha256((OUT/'texts.json').read_bytes()).hexdigest(),'challenge_sha256':hashlib.sha256((ROOT/'eval/quality/corpus/bilingual-hard-round2.json').read_bytes()).hexdigest()}
    torch.save(data,OUT/'data.pt');(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['supervised_counts','excluded_overlap','texts_sha256','challenge_sha256']},indent=2),flush=True)

if __name__=='__main__':prepare()
