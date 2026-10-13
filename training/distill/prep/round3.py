"""Round-three curriculum: scope, quantities, and topic-aware retrieval negatives."""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'training/distill'))
from tokenization import load_tokenizer
from prep.round2 import encode_rows, normalize, number

OUT = ROOT / 'training/distill/cache/mini-fa-en-round3'
PREVIOUS = ROOT / 'training/distill/cache/mini-fa-en-round2'
CHALLENGE = ROOT / 'eval/quality/corpus/bilingual-hard-round3.json'


def numeral(n, lang, style):
    if style == 0:
        if n < 0:
            return ('minus ' if lang == 'en' else 'منفی ') + numeral(-n, lang, style)
        if n >= 100:
            h, rest = divmod(n, 100)
            head = number(h, 'en') + ' hundred' if lang == 'en' else ['', 'صد', 'دویست', 'سیصد', 'چهارصد', 'پانصد', 'ششصد', 'هفتصد', 'هشتصد', 'نهصد'][h]
            return head + ((' and ' if lang == 'en' else ' و ') + number(rest, lang) if rest else '')
        return number(n, lang)
    return str(n).translate(str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')) if lang == 'fa' and style == 2 else str(n)


def synthetic(split, count=12000):
    rng = random.Random(301 if split == 'train' else 302)
    result = []
    objects = [('درخواست خرید', 'purchase request'), ('فرم مرخصی', 'leave form'), ('رزرو بلیت', 'ticket reservation'), ('پیشنهاد همکاری', 'partnership proposal'), ('درخواست عضویت', 'membership request'), ('پرونده استخدام', 'employment application')]
    if split == 'val':
        objects = [('درخواست بورسیه', 'scholarship application'), ('فرم بیمه', 'insurance form'), ('پرونده مجوز', 'permit application')]
    items = [('کتاب', 'books'), ('جعبه', 'boxes'), ('چراغ', 'lamps'), ('مداد', 'pencils'), ('کارت', 'cards'), ('صندلی', 'chairs')]
    if split == 'val':
        items = [('بطری', 'bottles'), ('دفتر', 'notebooks'), ('قطعه', 'parts')]
    names = [('علی', 'Alice'), ('مینا', 'Bob'), ('زهرا', 'David'), ('رضا', 'Emma'), ('نسرین', 'Paul'), ('حامد', 'Grace')]
    if split == 'val':
        names = [('پریسا', 'Arthur'), ('کیان', 'Diana'), ('الهام', 'George')]
    for lang in ['fa', 'en']:
        li = int(lang == 'en')
        for i in range(count):
            cat = ['negation', 'numbers', 'role-order'][i % 3]
            obj = rng.choice(objects)[li]; item = rng.choice(items)[li]
            a, b = [p[li] for p in rng.sample(names, 2)]
            style = rng.randrange(3); fmt = lambda n: numeral(n, lang, style)
            t = rng.randrange(8); positive = bool(rng.randrange(2))
            if cat == 'negation':
                if lang == 'fa':
                    yes = f'حداقل یک {obj} تایید شده است.'; no = f'هیچ {obj} تایید نشده است.'
                    if t == 0:
                        q = f'درست نیست که هیچ {obj} تایید نشده باشد.' if positive else f'هیچ‌کدام از {obj}ها تایید نشده‌اند.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 1:
                        q = f'این ادعا که همهٔ {obj}ها رد شده‌اند نادرست است.' if positive else f'تک‌تک {obj}ها رد شده‌اند و مورد تاییدشده‌ای وجود ندارد.'
                        yes, no = f'حداقل یک {obj} رد نشده است.', f'تمام {obj}ها رد شده‌اند.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 2:
                        q = f'همهٔ {obj}ها تایید نشده‌اند.' if positive else f'هیچ {obj} رد نشده و همه تایید شده‌اند.'
                        yes, no = f'حداقل یک {obj} تایید نشده است.', f'تمام {obj}ها تایید شده‌اند.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 3:
                        q = f'{a} متعهد شد {obj} را لغو نکند.' if positive else f'{a} متعهد شد {obj} را لغو کند.'
                        yes, no = f'{a} قول حفظ {obj} را داد.', f'{a} قول لغو {obj} را داد.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 4:
                        q = f'{a} دیگر {obj} ثبت نمی‌کند؛ قبلاً این کار را می‌کرد.' if positive else f'{a} هنوز هیچ {obj} ثبت نکرده است؛ قرار است شروع کند.'
                        yes, no = f'ثبت {obj} توسط {a} متوقف شده است.', f'ثبت {obj} توسط {a} هنوز آغاز نشده است.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 5:
                        q = f'نه درستی {obj} با مدرک ثابت شده و نه نادرستی آن؛ نتیجه هنوز نامعلوم است.' if positive else f'با مدرک قطعی ثابت شده که {obj} غلط است.'
                        yes, no = f'نادرستی {obj} هنوز ثابت نشده است.', f'نادرستی {obj} قطعاً ثابت شده است.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 6:
                        q = f'به جز {a}، همه {obj} را امضا کردند.' if positive else f'به جز {a}، هیچ‌کس {obj} را امضا نکرد.'
                        yes, no = f'{a} {obj} را امضا نکرده است.', f'{a} {obj} را امضا کرده است.'
                        p, n = (yes, no) if positive else (no, yes)
                    else:
                        q = f'تنها شرط مجاز بودن ارسال {obj}، امضا بودن آن است. این {obj} امضا شده است.' if positive else f'تنها شرط مجاز بودن ارسال {obj}، امضا بودن آن است. این {obj} امضا نشده است.'
                        yes, no = f'ارسال {obj} مجاز است.', f'ارسال {obj} مجاز نیست.'
                        p, n = (yes, no) if positive else (no, yes)
                else:
                    yes = f'At least one {obj} has been approved.'; no = f'No {obj} has been approved.'
                    if t == 0:
                        q = f'It is untrue that no {obj} has been approved.' if positive else f'Not a single {obj} has been approved.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 1:
                        q = f'The claim that every {obj} was rejected is false.' if positive else f'Every {obj} was rejected without exception.'
                        yes, no = f'At least one {obj} was not rejected.', f'Every {obj} was rejected.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 2:
                        q = f'Not every {obj} has been approved.' if positive else f'No {obj} was rejected and all were approved.'
                        yes, no = f'At least one {obj} has not been approved.', f'Every {obj} has been approved.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 3:
                        q = f'{a} promised not to cancel the {obj}.' if positive else f'{a} promised to cancel the {obj}.'
                        yes, no = f'{a} promised to keep the {obj}.', f'{a} promised to cancel the {obj}.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 4:
                        q = f'{a} no longer files a {obj}, although previously they did.' if positive else f'{a} has not yet filed any {obj}; they plan to start.'
                        yes, no = f'{a} has stopped filing a {obj}.', f'{a} has not started filing a {obj}.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 5:
                        q = f'Neither the correctness nor the incorrectness of the {obj} has been proven; the result remains unknown.' if positive else f'Conclusive evidence proves the {obj} is wrong.'
                        yes, no = f'The {obj} has not been proven wrong.', f'The {obj} has definitely been proven wrong.'
                        p, n = (yes, no) if positive else (no, yes)
                    elif t == 6:
                        q = f'Everyone except {a} signed the {obj}.' if positive else f'Nobody except {a} signed the {obj}.'
                        yes, no = f'{a} did not sign the {obj}.', f'{a} signed the {obj}.'
                        p, n = (yes, no) if positive else (no, yes)
                    else:
                        q = f'The only condition for permission to send the {obj} is a signature. This {obj} is signed.' if positive else f'The only condition for permission to send the {obj} is a signature. This {obj} is not signed.'
                        yes, no = f'The {obj} may be sent.', f'The {obj} may not be sent.'
                        p, n = (yes, no) if positive else (no, yes)
                topic = 'negation-' + obj
            elif cat == 'numbers':
                x, y = rng.sample(range(2, 301), 2)
                if t == 0:
                    total, used = max(x, y), min(x, y); left = total - used
                    if lang == 'fa':
                        q = f'از {fmt(total)} {item}، {fmt(used)} {item} هنوز تحویل داده نشده است. چند {item} تحویل داده شده؟'
                        p, n = f'{fmt(left)} {item} تحویل داده شده است.', f'{fmt(used)} {item} تحویل داده شده است.'
                    else:
                        q = f'Of {fmt(total)} {item}, {fmt(used)} have not been delivered. How many have been delivered?'
                        p, n = f'{fmt(left)} {item} have been delivered.', f'{fmt(used)} {item} have been delivered.'
                elif t == 1:
                    x = rng.choice(range(100, 801, 20)); pct = rng.choice([5, 10, 20, 25, 50]); answer = x * (100-pct) // 100
                    if lang == 'fa':
                        q = f'قیمت {item} {fmt(x)} است و {fmt(pct)} درصد تخفیف دارد.'
                        p, n = f'قیمت نهایی {item} برابر {fmt(answer)} است.', f'قیمت نهایی {item} برابر {fmt(x-pct)} است.'
                    else:
                        q = f'The price of {item} is {fmt(x)} with a {fmt(pct)} percent discount.'
                        p, n = f'The final price of {item} is {fmt(answer)}.', f'The final price of {item} is {fmt(x-pct)}.'
                elif t == 2:
                    x = rng.randrange(12, 80); y = x + rng.choice([-2, -1, 0, 1, 2]); strict = bool(rng.randrange(2)); ok = y > x if strict else y >= x
                    if lang == 'fa':
                        q = f'برای پذیرش، سن باید {"بیشتر از" if strict else "حداقل"} {fmt(x)} سال باشد. سن {a} {fmt(y)} سال است.'
                        yes, no = f'{a} شرط سنی پذیرش را دارد.', f'{a} شرط سنی پذیرش را ندارد.'
                    else:
                        q = f'Admission requires an age {"greater than" if strict else "of at least"} {fmt(x)}. {a} is {fmt(y)} years old.'
                        yes, no = f'{a} meets the admission age requirement.', f'{a} does not meet the admission age requirement.'
                    p, n = (yes, no) if ok else (no, yes)
                elif t == 3:
                    x, y = rng.sample(range(-20, 40), 2); delta = abs(y-x)
                    if lang == 'fa':
                        q = f'دمای انبار {item} از {fmt(x)} به {fmt(y)} درجه رسید.'
                        yes, no = f'انبار {item} {fmt(delta)} درجه گرم‌تر شد.', f'انبار {item} {fmt(delta)} درجه سردتر شد.'
                    else:
                        q = f'The temperature in the warehouse for {item} moved from {fmt(x)} to {fmt(y)} degrees.'
                        yes, no = f'The warehouse for {item} became {fmt(delta)} degrees warmer.', f'The warehouse for {item} became {fmt(delta)} degrees colder.'
                    p, n = (yes, no) if y > x else (no, yes)
                elif t == 4:
                    x, y = rng.sample(range(5, 96, 5), 2); delta = abs(y-x)
                    if lang == 'fa':
                        q = f'درصد {item} سالم از {fmt(x)} به {fmt(y)} رسیده است.'
                        yes, no = f'سهم {item} سالم {fmt(delta)} واحد درصد بیشتر شده است.', f'سهم {item} سالم {fmt(delta)} واحد درصد کمتر شده است.'
                    else:
                        q = f'The percentage of undamaged {item} moved from {fmt(x)} to {fmt(y)}.'
                        yes, no = f'The share of undamaged {item} increased by {fmt(delta)} percentage points.', f'The share of undamaged {item} decreased by {fmt(delta)} percentage points.'
                    p, n = (yes, no) if y > x else (no, yes)
                elif t == 5:
                    total, used = max(x, y), min(x, y); returned = rng.randrange(1, 20); answer = total-used+returned
                    if lang == 'fa':
                        q = f'موجودی {fmt(total)} {item} بود؛ {fmt(used)} فروختیم و {fmt(returned)} مرجوع شد.'
                        p, n = f'موجودی {item} اکنون {fmt(answer)} است.', f'موجودی {item} اکنون {fmt(total-used-returned)} است.'
                    else:
                        q = f'We stocked {fmt(total)} {item}, sold {fmt(used)}, and received {fmt(returned)} returns.'
                        p, n = f'The current stock of {item} is {fmt(answer)}.', f'The current stock of {item} is {fmt(total-used-returned)}.'
                elif t == 6:
                    hour = rng.randrange(6, 20); minute = rng.randrange(30, 60); duration = rng.randrange(15, 100); end = hour*60+minute+duration
                    start_time=f'{hour}:{minute:02d}';end_time=f'{end//60}:{end%60:02d}'
                    if lang == 'fa':
                        q = f'بسته‌بندی {item} ساعت {start_time} شروع و ساعت {end_time} تمام شد.'
                        p, n = f'بسته‌بندی {item} {fmt(duration)} دقیقه طول کشید.', f'بسته‌بندی {item} {fmt(duration+40)} دقیقه طول کشید.'
                    else:
                        q = f'Packing the {item} started at {start_time} and ended at {end_time}.'
                        p, n = f'Packing the {item} took {fmt(duration)} minutes.', f'Packing the {item} took {fmt(duration+40)} minutes.'
                else:
                    x=rng.randrange(1, 90);y=x+rng.choice([-1,0,1]);strict=bool(rng.randrange(2));ok=y<x if strict else y<=x
                    # Tenths retain exact comparisons without floating-point label arithmetic.
                    fdec=lambda z: f'{z//10}.{z%10}'.translate(str.maketrans('0123456789.', '۰۱۲۳۴۵۶۷۸۹٫')) if lang=='fa' and style==2 else f'{z/10:.1f}'
                    if lang=='fa':
                        q=f'وزن {item} باید {"کمتر از" if strict else "حداکثر"} {fdec(x)} کیلو باشد؛ وزن فعلی {fdec(y)} کیلو است.'
                        yes,no=f'وزن {item} در محدودهٔ مجاز است.',f'وزن {item} در محدودهٔ مجاز نیست.'
                    else:
                        q=f'The weight of {item} must be {"less than" if strict else "at most"} {fdec(x)} kilograms; the actual weight is {fdec(y)} kilograms.'
                        yes,no=f'The weight of {item} is within the permitted range.',f'The weight of {item} is outside the permitted range.'
                    p,n=(yes,no) if ok else (no,yes)
                topic = 'numbers-' + item
            else:
                if lang == 'fa':
                    q=f'{b} توسط {a} برای بررسی {obj} دعوت شد.'
                    p,n=f'{a} {b} را دعوت کرد.',f'{b} {a} را دعوت کرد.'
                else:
                    q=f'{b} was invited by {a} to review the {obj}.'
                    p,n=f'{a} invited {b}.',f'{b} invited {a}.'
                topic='role-order'
            if normalize(p) != normalize(n):
                result.append(dict(anchor=q,positive=p,negative=n,lang=lang,category=cat,topic=topic,template=t))
    return list({(r['anchor'],r['positive'],r['negative']):r for r in result}.values())


def prepare():
    if OUT.exists():
        raise FileExistsError('Preserve prepared round-three data')
    old = json.loads((PREVIOUS/'texts.json').read_text())
    manifest = json.loads((PREVIOUS/'manifest.json').read_text())
    tokenizer = load_tokenizer(manifest['tokenizer'])
    held = json.loads(CHALLENGE.read_text())
    forbidden = {normalize(r['text']) for r in held['corpus']+held['queries']}
    rows = {}
    for split in ['val','train']:
        candidates = synthetic(split, 600 if split=='val' else 18000)
        nli = [dict(r,topic='nli-'+str(i),template=-1) for i,r in enumerate(old['triplets'][split]) if r['category']=='nli']
        candidates += nli
        rows[split] = [r for r in candidates if all(normalize(r[k]) not in forbidden for k in ['anchor','positive','negative'])]
        if split=='val':
            forbidden.update(normalize(r[k]) for r in rows[split] for k in ['anchor','positive','negative'])
    assert min(len(v) for v in rows.values()) > 500
    data = {}
    topics = {t:i for i,t in enumerate(sorted({r['topic'] for v in rows.values() for r in v}))}
    docs = {t:i for i,t in enumerate(sorted({normalize(r[k]) for v in rows.values() for r in v for k in ['positive','negative']}))}
    for split, samples in rows.items():
        batch = encode_rows(samples,['anchor','positive','negative'],tokenizer)
        batch['group']=torch.tensor([6 if r['category']=='nli' else (0 if r['lang']=='fa' else 3)+['negation','numbers','role-order'].index(r['category']) for r in samples])
        batch['topic']=torch.tensor([topics[r['topic']] for r in samples])
        batch['doc_ids']=torch.tensor([[docs[normalize(r[k])] for k in ['positive','negative']] for r in samples])
        data[split]=batch
    OUT.mkdir(parents=True)
    (OUT/'texts.json').write_text(json.dumps(rows,ensure_ascii=False)+'\n')
    manifest.update(round=3,challenge_sha256=hashlib.sha256(CHALLENGE.read_bytes()).hexdigest(),texts_sha256=hashlib.sha256((OUT/'texts.json').read_bytes()).hexdigest(),ranking_counts={k:len(v) for k,v in rows.items()},selection_note='Round-three full-corpus validation and STS gates only; no final-test selection')
    torch.save(data,OUT/'data.pt')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['ranking_counts']),flush=True)


if __name__=='__main__':
    prepare()
