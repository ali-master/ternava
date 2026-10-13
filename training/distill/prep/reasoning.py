"""Counterfactual supervision with disjoint wording, nouns and numeric ranges."""
import json
import random

import torch

from prep.native import ROOT, digest
from prep.round2 import encode_rows, normalize
from prep.round4 import negation, render_number
from tokenization import load_tokenizer

CACHE=ROOT/'training/distill/cache/mini-fa-en-reasoning'
SOURCE=ROOT/'training/distill/runs/mini-fa-en-domain/diagnostic-best.pt'
CHALLENGE=ROOT/'eval/quality/corpus/persian-reasoning-hard.json'


def numeric(rng, obj, lang, family, variant, validation=False):
    # Labels use exact integer arithmetic here only; inference never sees this generator.
    x=rng.randrange(401,501) if validation else rng.randrange(20,301)
    y=rng.randrange(2,19);z=rng.randrange(1,10)
    fmt=lambda n:render_number(n,lang,rng.randrange(3))
    a,b,c=fmt(x),fmt(y),fmt(z)
    if family==0:
        values=[x+y,x-y]
        if lang=='fa':
            templates=[(f'{obj}: موجودی {a}، ورودی {b}. حاصل نهایی؟',f'{obj}: موجودی {a}، خروجی {b}. حاصل نهایی؟'),
                       (f'{a} {obj} داشتیم و {b} مورد دریافت کردیم.',f'{a} {obj} داشتیم و {b} مورد تحویل دادیم.'),
                       (f'تعداد {obj}ها {a} بود. {b} مورد به آن افزودند.',f'تعداد {obj}ها {a} بود. {b} مورد از آن کاستند.'),
                       (f'{b} مورد به موجودی قبلی {a}تایی {obj} وارد شد.',f'{b} مورد از موجودی قبلی {a}تایی {obj} خارج شد.')]
        else:
            templates=[(f'{obj}: stock {a}, incoming {b}. Final result?',f'{obj}: stock {a}, outgoing {b}. Final result?'),
                       (f'We had {a} {obj}s and received {b}.',f'We had {a} {obj}s and gave away {b}.'),
                       (f'The {obj} count was {a}. Another {b} were added.',f'The {obj} count was {a}. Another {b} were removed.'),
                       (f'{b} entered the existing stock of {a} {obj}s.',f'{b} left the existing stock of {a} {obj}s.')]
    elif family==1:
        values=[x-y+z,x-y-z]
        if lang=='fa':
            templates=[(f'{obj}: شروع {a}، کاهش {b}، افزایش {c}. پایان؟',f'{obj}: شروع {a}، کاهش {b}، کاهش {c}. پایان؟'),
                       (f'{a} {obj} را با کم کردن {b} و سپس افزودن {c} بشمار.',f'{a} {obj} را با کم کردن {b} و سپس کم کردن {c} بشمار.'),
                       (f'موجودی {obj} برابر {a} بود؛ {b} فروختیم و {c} خریدیم.',f'موجودی {obj} برابر {a} بود؛ {b} فروختیم و {c} دیگر فروختیم.'),
                       (f'ابتدا {b} از {a} {obj} بردار؛ بعد {c} را اضافه کن.',f'ابتدا {b} از {a} {obj} بردار؛ بعد {c} را هم بردار.')]
        else:
            templates=[(f'{obj}: start {a}, subtract {b}, add {c}. End?',f'{obj}: start {a}, subtract {b}, subtract {c}. End?'),
                       (f'Count {a} {obj}s after removing {b} and then adding {c}.',f'Count {a} {obj}s after removing {b} and then removing {c}.'),
                       (f'We stocked {a} {obj}s; sold {b} and bought {c}.',f'We stocked {a} {obj}s; sold {b} and sold another {c}.'),
                       (f'First take {b} from {a} {obj}s; then add {c}.',f'First take {b} from {a} {obj}s; then take {c} too.')]
    elif family==2:
        x=(x//20)*20;pct=rng.choice([10,20,25,50]);a,b=fmt(x),fmt(pct);values=[x*(100-pct)//100,x*(100+pct)//100]
        if lang=='fa':
            templates=[(f'{obj} با قیمت {a}، {b} درصد ارزان شد. بهای تازه؟',f'{obj} با قیمت {a}، {b} درصد گران شد. بهای تازه؟'),
                       (f'قیمت {obj} را از {a}، {b} درصد پایین بیاور.',f'قیمت {obj} را از {a}، {b} درصد بالا ببر.'),
                       (f'برای {obj} به ارزش {a}، {b} درصد تخفیف حساب کن.',f'برای {obj} به ارزش {a}، {b} درصد اضافه‌بها حساب کن.'),
                       (f'بهای قبلی {obj} برابر {a} است؛ کاهش نسبی {b} درصدی اعمال کن.',f'بهای قبلی {obj} برابر {a} است؛ رشد نسبی {b} درصدی اعمال کن.')]
        else:
            templates=[(f'A {obj} costing {a} became {b} percent cheaper. New price?',f'A {obj} costing {a} became {b} percent dearer. New price?'),
                       (f'Lower the {obj} price of {a} by {b} percent.',f'Raise the {obj} price of {a} by {b} percent.'),
                       (f'Apply a {b} percent discount to a {obj} valued at {a}.',f'Apply a {b} percent surcharge to a {obj} valued at {a}.'),
                       (f'The old {obj} price is {a}; apply a relative decrease of {b} percent.',f'The old {obj} price is {a}; apply a relative increase of {b} percent.')]
    elif family==3:
        den=rng.choice([3,5,7,8]);part=rng.randrange(1,den)
        if part*2==den:part=1
        total=(x//den)*den;values=[total//den*part,total//den*(den-part)];a,b,c=fmt(total),fmt(part),fmt(den)
        if lang=='fa':
            templates=[(f'{a} {obj} داریم و {b} از هر {c} مورد سالم‌اند. چند سالم؟',f'{a} {obj} داریم و {b} از هر {c} مورد سالم‌اند. چند خراب؟'),
                       (f'سهم سالم از {a} {obj}، {b}/{c} است. سالم‌ها را بشمار.',f'سهم سالم از {a} {obj}، {b}/{c} است. بقیه را بشمار.'),
                       (f'از کل {a} {obj} کسر {b}/{c} سالم است. تعداد سالم چیست؟',f'از کل {a} {obj} کسر {b}/{c} سالم است. تعداد ناسالم چیست؟'),
                       (f'برای مجموعهٔ {a}تایی {obj}، نسبت سالم {b} به {c} است؛ شمار سالم؟',f'برای مجموعهٔ {a}تایی {obj}، نسبت سالم {b} به {c} است؛ شمار مکمل؟')]
        else:
            templates=[(f'We have {a} {obj}s and {b} of every {c} are sound. How many sound?',f'We have {a} {obj}s and {b} of every {c} are sound. How many defective?'),
                       (f'The sound share of {a} {obj}s is {b}/{c}. Count the sound ones.',f'The sound share of {a} {obj}s is {b}/{c}. Count the remainder.'),
                       (f'Of all {a} {obj}s, a fraction {b}/{c} is sound. Sound count?',f'Of all {a} {obj}s, a fraction {b}/{c} is sound. Defective count?'),
                       (f'For a set of {a} {obj}s the sound ratio is {b} to {c}; sound count?',f'For a set of {a} {obj}s the sound ratio is {b} to {c}; complementary count?')]
    else:
        # Reuse the signed/boundary/time families but vary the numeric seed and domain nouns.
        from prep.round4 import numeric as existing
        return existing(rng,obj,lang,[4,5,7][family-4],variant)
    q0,q1=templates[variant]
    answers=[f'حاصل برای {obj} برابر {fmt(v)} است.' if lang=='fa' else f'The result for the {obj} is {fmt(v)}.' for v in values]
    return q0,q1,*answers


def synthetic(split,count=1400):
    rng=random.Random(719 if split=='train' else 720);rows=[]
    nouns=[('جعبه','box'),('مهره','bead'),('دکمه','button'),('سکه','coin'),('پاکت','envelope'),('بسته','packet')]
    if split=='val':nouns=[('قرقره','spool'),('گیره','clip')]
    for lang in ['fa','en']:
        for i in range(count):
            obj=rng.choice(nouns)[lang=='en'];variant=3 if split=='val' else rng.randrange(3)
            for category in ['negation','numbers']:
                if category=='numbers':pair=numeric(rng,obj,lang,i%7,variant,split=='val')
                elif i%10<8:pair=negation(obj,'مهتاب' if lang=='fa' else 'Doris',lang,i%10,variant)
                elif i%10==8:
                    pair=(f'وجود مجوز برای انتقال {obj} لازم است. مجوز داریم؛ دیگر شرط‌ها نامعلوم‌اند.',f'وجود مجوز برای انتقال {obj} لازم است. مجوز نداریم و استثنایی نیست.',f'مجاز بودن انتقال {obj} نامعلوم است.',f'انتقال {obj} مجاز نیست.') if lang=='fa' else (f'A permit is required to transfer the {obj}. We have one; other conditions are unknown.',f'A permit is required to transfer the {obj}. We lack one and there are no exceptions.',f'Permission to transfer the {obj} is unknown.',f'Transfer of the {obj} is forbidden.')
                    if split=='val':pair=(f'اجازهٔ جابه‌جایی {obj} به مجوز وابسته است؛ مجوز فراهم است، باقی الزامات مشخص نیست.',f'اجازهٔ جابه‌جایی {obj} به مجوز وابسته است؛ بدون هیچ استثنا مجوز مفقود است.',pair[2],pair[3]) if lang=='fa' else (f'Moving the {obj} depends on a permit; it exists but further requirements are unspecified.',f'Moving the {obj} depends on a permit; it is missing without exception.',pair[2],pair[3])
                else:
                    pair=(f'هیچ مدرکی برای خراب بودن {obj} نداریم؛ سالم بودن هم ثابت نیست.',f'مدرک قطعی داریم که {obj} خراب نیست.',f'خرابی {obj} رد نشده است.',f'خرابی {obj} رد شده است.') if lang=='fa' else (f'There is no evidence the {obj} is defective; soundness is not proven either.',f'Conclusive evidence proves the {obj} is not defective.',f'A defect in the {obj} has not been ruled out.',f'A defect in the {obj} has been ruled out.')
                    if split=='val':pair=(f'دربارهٔ خرابی {obj} شواهدی ارائه نشده؛ از صحت آن هم اطمینانی نداریم.',f'بررسی قطعی، سالم بودن {obj} را تایید می‌کند.',pair[2],pair[3]) if lang=='fa' else (f'No testimony establishes a defect in the {obj}; its soundness is also uncertain.',f'A definitive examination confirms the {obj} is sound.',pair[2],pair[3])
                if pair is None:continue
                q0,q1,p,n=pair
                for flip,(q,a,b) in enumerate([(q0,p,n),(q1,n,p)]):rows.append(dict(anchor=q,positive=a,negative=b,category=category,lang=lang,pair=f'{lang}-{category}-{i}',flip=flip))
    unique={}
    for i in range(0,len(rows),2):unique[tuple(rows[j][k] for j in [i,i+1] for k in ['anchor','positive','negative'])]=rows[i:i+2]
    return [r for pair in unique.values() for r in pair]


def prepare():
    if CACHE.exists():raise FileExistsError('Preserve reasoning data')
    source=torch.load(SOURCE,map_location='cpu',weights_only=False);tk=load_tokenizer(source['tokenizer'])
    fresh=json.loads(CHALLENGE.read_text());forbidden={normalize(q['text']) for f in fresh['suites'].values() for q in f['queries']+([] if f in [fresh['suites']['docs'],fresh['suites']['emoji']] else f['corpus'])}
    data={};texts={}
    for split in ['val','train']:
        rows=synthetic(split,140 if split=='val' else 1400);batch=encode_rows(rows,['anchor','positive','negative'],tk);keep=[]
        for i in range(0,len(rows),2):
            if all(not torch.equal(batch['input_ids'][j,1],batch['input_ids'][j,2]) and all(normalize(rows[j][k]) not in forbidden for k in ['anchor','positive','negative']) for j in [i,i+1]):keep.extend([i,i+1])
        texts[split]=[rows[i] for i in keep];data[split]={k:v[keep] for k,v in batch.items()}
        data[split]['group']=torch.tensor([(0 if r['lang']=='fa' else 2)+(r['category']=='numbers') for r in texts[split]])
        forbidden.update(normalize(r[k]) for r in texts[split] for k in ['anchor','positive','negative'])
    CACHE.mkdir(parents=True);torch.save(data,CACHE/'data.pt')
    (CACHE/'texts.json').write_text(json.dumps(texts,ensure_ascii=False)+'\n')
    manifest=dict(source_sha256=digest(SOURCE),challenge_sha256=digest(CHALLENGE),tokenizer_sha256=digest(source['tokenizer']['path']),counts={k:len(v) for k,v in texts.items()},files={p.name:digest(p) for p in CACHE.iterdir()})
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)


if __name__=='__main__':prepare()
