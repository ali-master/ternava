"""Counterfactual pairs with separate validation wording, fixed round-three tokenizer."""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'training/distill'))
from prep.round2 import encode_rows, normalize
from prep.round3 import numeral
from tokenization import load_tokenizer

OUT = ROOT/'training/distill/cache/mini-fa-en-round4'
PREVIOUS = ROOT/'training/distill/cache/mini-fa-en-round2'
ROUND3 = ROOT/'training/distill/cache/mini-fa-en-round3'
CHALLENGE = ROOT/'eval/quality/corpus/bilingual-hard-round4.json'


def negation(obj, a, lang, family, variant):
    """Return two opposite queries with the same two answers; no closed-world inference."""
    if lang == 'fa':
        data = [
            ([f'ادعای نبودن حتی یک {obj} سالم غلط است.', f'این گفته نادرست است: هیچ {obj} سالمی وجود ندارد.', f'نبود هرگونه {obj} سالم را تکذیب می‌کنم.', f'آیا هیچ {obj} سالمی نداریم؟ خیر، این ادعا رد شده است.'],
             [f'حتی یک {obj} سالم نداریم.', f'در میان {obj}ها مورد سالمی نیست.', f'نبود هرگونه {obj} سالم تایید شد.', f'آیا {obj} سالمی داریم؟ خیر، هیچ موردی نداریم.'], f'بعضی {obj}ها سالم‌اند.', f'هیچ {obj} سالم نیست.'),
            ([f'درست نیست که تمام {obj}ها تایید شده باشند.', f'ادعای تایید همهٔ {obj}ها رد شده است.', f'تایید شدن تک‌تک {obj}ها حقیقت ندارد.', f'تایید عمومی {obj}ها استثنا دارد؛ دست‌کم یکی تایید نشده.'],
             [f'تمام {obj}ها بدون استثنا تایید شدند.', f'هیچ {obj} تاییدنشده‌ای باقی نمانده است.', f'تک‌تک {obj}ها تایید شده‌اند.', f'در تایید {obj}ها استثنایی وجود ندارد.'], f'دست‌کم یک {obj} تایید نشده است.', f'همهٔ {obj}ها تایید شده‌اند.'),
            ([f'{a} قول داد {obj} را از بین نبرد.', f'{a} تعهد کرد نابود کردن {obj} را انجام ندهد.', f'{a} تضمین کرد {obj} را حفظ کند.', f'تعهد {a} دربارهٔ {obj}، خودداری از نابود کردن آن بود.'],
             [f'{a} قول داد {obj} را از بین ببرد.', f'{a} تعهد کرد نابود کردن {obj} را انجام دهد.', f'{a} تضمین کرد {obj} را نابود کند.', f'تعهد {a} دربارهٔ {obj}، نابود کردن آن بود.'], f'{a} وعدهٔ حفظ {obj} را داد.', f'{a} وعدهٔ نابودی {obj} را داد.'),
            ([f'تولید {obj} دیگر انجام نمی‌شود؛ قبلاً فعال بود.', f'کار ساخت {obj} پس از سال‌ها متوقف شد.', f'ساخت {obj} را ادامه نمی‌دهیم؛ قبلاً می‌ساختیم.', f'زمانی {obj} تولید می‌کردیم، اما اکنون از آن دست کشیده‌ایم.'],
             [f'تولید {obj} هنوز شروع نشده؛ قبلاً هم نداشتیم.', f'کار ساخت {obj} از ابتدا آغاز نشده است.', f'ساخت {obj} را شروع نکرده‌ایم؛ قبلاً هم نساختیم.', f'ساخت {obj} در برنامه است ولی تا این لحظه هرگز آغاز نشده.'], f'تولید {obj} متوقف شده است.', f'تولید {obj} هنوز آغاز نشده است.'),
            ([f'نه سالم بودن {obj} ثابت شده نه خراب بودن آن.', f'برای سلامت یا خرابی {obj} مدرک قطعی نداریم.', f'وضعیت {obj} نامعلوم است؛ نمی‌دانیم سالم است یا نه.', f'بررسی {obj} نتیجه‌ای نداده؛ هر دو احتمال سلامت و خرابی باز است.'],
             [f'سالم بودن {obj} با مدرک قطعی ثابت شده است.', f'برای سلامت {obj} مدرک قطعی داریم.', f'وضعیت {obj} معلوم است؛ قطعاً سالم است.', f'بررسی {obj} تمام شده و سلامت آن به اثبات رسیده است.'], f'سلامت {obj} هنوز اثبات نشده است.', f'سلامت {obj} اثبات شده است.'),
            ([f'همه به جز {a}، {obj} را دیدند؛ {a} ندید.', f'{a} تنها کسی بود که {obj} را ندید.', f'تنها استثنا در دیدن {obj}، {a} بود که آن را ندید.', f'از جمع بازدیدکنندگان {obj} فقط {a} غایب بود.'],
             [f'هیچ‌کس به جز {a}، {obj} را ندید؛ {a} دید.', f'{a} تنها کسی بود که {obj} را دید.', f'تنها استثنا در ندیدن {obj}، {a} بود که آن را دید.', f'از جمع فقط {a} به بازدید {obj} رفت.'], f'{a} {obj} را ندید.', f'{a} {obj} را دید.'),
            ([f'نمی‌گویم {obj} نمی‌تواند خراب باشد.', f'احتمال خراب بودن {obj} را رد نمی‌کنم.', f'خرابی {obj} غیرممکن نیست.', f'این ادعا که {obj} محال است خراب باشد پذیرفته نیست؛ شاید خراب باشد.'],
             [f'می‌گویم {obj} نمی‌تواند خراب باشد.', f'هرگونه احتمال خراب بودن {obj} را رد می‌کنم.', f'خرابی {obj} غیرممکن است.', f'این ادعا که {obj} محال است خراب باشد قطعی است.'], f'خرابی {obj} ممکن است.', f'خرابی {obj} محال است.'),
            ([f'دستور توقف ساخت {obj} لغو نشد و همچنان برقرار است.', f'توقف ساخت {obj} را برنداشتیم؛ ساخت ممنوع مانده است.', f'ساخت {obj} ممنوع است؛ لغو ممنوعیت صورت نگرفته.', f'تصمیم به توقف ساخت {obj} هنوز معتبر است و بازگشتی رخ نداده.'],
             [f'دستور توقف ساخت {obj} لغو شد و ساخت از سر گرفته شد.', f'توقف ساخت {obj} را برداشتیم؛ ساخت شروع شده است.', f'ساخت {obj} آغاز شد؛ ممنوعیت لغو شده.', f'تصمیم به توقف ساخت {obj} دیگر معتبر نیست و ساخت دوباره شروع شد.'], f'ساخت {obj} متوقف است.', f'ساخت {obj} از سر گرفته شده است.'),
        ]
    else:
        data = [
            ([f'The claim that no sound {obj} exists is false.', f'It is untrue that there is not a single sound {obj}.', f'I deny the absence of any sound {obj}.', f'Are there no sound {obj}s? No, that claim has been refuted.'],
             [f'Not a single sound {obj} exists.', f'There are no sound {obj}s at all.', f'The absence of any sound {obj} is confirmed.', f'Do any sound {obj}s exist? No, none do.'], f'Some {obj}s are sound.', f'No {obj} is sound.'),
            ([f'It is not true that all {obj}s were approved.', f'The claim that every {obj} was approved has been denied.', f'Approval of every single {obj} is not a fact.', f'The blanket approval of {obj}s has exceptions; at least one was not approved.'],
             [f'Every {obj} was approved without exception.', f'No unapproved {obj} remains.', f'Each and every {obj} was approved.', f'There are no exceptions to the approval of the {obj}s.'], f'At least one {obj} was not approved.', f'All {obj}s were approved.'),
            ([f'{a} promised not to destroy the {obj}.', f'{a} undertook to refrain from destroying the {obj}.', f'{a} guaranteed the preservation of the {obj}.', f'The commitment {a} made about the {obj} was to avoid its destruction.'],
             [f'{a} promised to destroy the {obj}.', f'{a} undertook to destroy the {obj}.', f'{a} guaranteed the destruction of the {obj}.', f'The commitment {a} made about the {obj} was to destroy it.'], f'{a} promised to preserve the {obj}.', f'{a} promised to destroy the {obj}.'),
            ([f'We no longer produce the {obj}; we used to.', f'Production of the {obj} stopped after years.', f'We will not continue making the {obj}; we made it before.', f'There was a time when we produced the {obj}, but we have since given it up.'],
             [f'We have not yet started producing the {obj}; we never did.', f'Production of the {obj} has never begun.', f'We have not started making the {obj}; we never made it before.', f'Producing the {obj} is planned but has never begun up to this point.'], f'Production of the {obj} has stopped.', f'Production of the {obj} has not started.'),
            ([f'Neither the soundness nor the defectiveness of the {obj} has been proven.', f'There is no conclusive evidence that the {obj} is sound or defective.', f'The condition of the {obj} is unknown; we cannot say it is sound.', f'The examination of the {obj} was inconclusive; both soundness and defects remain possible.'],
             [f'The soundness of the {obj} has been conclusively proven.', f'There is conclusive evidence that the {obj} is sound.', f'The condition of the {obj} is known; it is certainly sound.', f'The examination of the {obj} is complete and establishes its soundness.'], f'The soundness of the {obj} is unproven.', f'The soundness of the {obj} is proven.'),
            ([f'Everyone except {a} saw the {obj}; {a} did not.', f'{a} was the only person who did not see the {obj}.', f'The sole exception to seeing the {obj} was {a}, who did not see it.', f'Among the visitors to the {obj}, only {a} was absent.'],
             [f'Nobody except {a} saw the {obj}; {a} did.', f'{a} was the only person who saw the {obj}.', f'The sole exception to not seeing the {obj} was {a}, who saw it.', f'Of the group, only {a} visited the {obj}.'], f'{a} did not see the {obj}.', f'{a} saw the {obj}.'),
            ([f'I do not say the {obj} cannot be defective.', f'I do not rule out a defect in the {obj}.', f'A defect in the {obj} is not impossible.', f'The assertion that a defective {obj} is impossible is unacceptable; it might be defective.'],
             [f'I say the {obj} cannot be defective.', f'I rule out any possibility of a defect in the {obj}.', f'A defect in the {obj} is impossible.', f'The assertion that a defective {obj} is impossible is certain.'], f'A defect in the {obj} is possible.', f'A defect in the {obj} is impossible.'),
            ([f'The suspension of making the {obj} was not lifted and remains in force.', f'We did not end the suspension of making the {obj}; production is still banned.', f'Making the {obj} is banned; the ban was not cancelled.', f'The decision to stop making the {obj} is still valid and has not been reversed.'],
             [f'The suspension of making the {obj} was lifted and production resumed.', f'We ended the suspension of making the {obj}; production has begun.', f'Making the {obj} has begun; the ban was cancelled.', f'The decision to stop making the {obj} is no longer valid and production restarted.'], f'Production of the {obj} is suspended.', f'Production of the {obj} has resumed.'),
        ]
    yes,no,p,n=data[family]
    return yes[variant],no[variant],p,n


def render_number(n, lang, style):
    if style == 0 and n >= 1000:
        head = numeral(n // 1000, lang, 0) + (' thousand' if lang == 'en' else ' هزار')
        return head + ((' and ' if lang == 'en' else ' و ') + numeral(n % 1000, lang, 0) if n % 1000 else '')
    return numeral(n, lang, style)


def numeric(rng, obj, lang, family, variant):
    """Counterfactual operations/relations share answers, with exact integer labels."""
    fmt=lambda n:render_number(n,lang,rng.randrange(3))
    x=rng.randrange(20,180);y=rng.randrange(2,19)
    if family in [0,1,2,3]:
        if family==0:
            answers=(x+y,x-y)
            fa=(f'موجودی {obj} {fmt(x)} بود؛ {fmt(y)} اضافه شد.',f'موجودی {obj} {fmt(x)} بود؛ {fmt(y)} کم شد.')
            en=(f'The stock of {obj}s was {fmt(x)}; {fmt(y)} were added.',f'The stock of {obj}s was {fmt(x)}; {fmt(y)} were removed.')
            if variant==3:
                fa=(f'{fmt(y)} {obj} به {fmt(x)} مورد قبلی پیوست. شمار نهایی؟',f'از {fmt(x)} {obj} قبلی، {fmt(y)} مورد برداشته شد. شمار نهایی؟')
                en=(f'{fmt(y)} {obj}s joined the previous {fmt(x)}. Final count?',f'From the previous {fmt(x)} {obj}s, {fmt(y)} were taken away. Final count?')
        elif family==1:
            x=rng.choice(range(120,801,20));pct=rng.choice([5,10,20,25,50]);answers=(x*(100-pct)//100,x*(100+pct)//100)
            fa=(f'بهای {obj} {fmt(x)} است؛ {fmt(pct)} درصد کاهش بده.',f'بهای {obj} {fmt(x)} است؛ {fmt(pct)} درصد افزایش بده.')
            en=(f'The price of the {obj} is {fmt(x)}; reduce it by {fmt(pct)} percent.',f'The price of the {obj} is {fmt(x)}; increase it by {fmt(pct)} percent.')
            if variant==3:
                fa=(f'{obj} به قیمت {fmt(x)} با تخفیف {fmt(pct)} درصد عرضه می‌شود.',f'{obj} به قیمت {fmt(x)} با افزایش {fmt(pct)} درصد عرضه می‌شود.')
                en=(f'A {obj} priced at {fmt(x)} is offered with a {fmt(pct)} percent discount.',f'A {obj} priced at {fmt(x)} is offered with a {fmt(pct)} percent markup.')
        elif family==2:
            z=rng.randrange(1,15);answers=(x-y+z,x-y-z)
            fa=(f'از {fmt(x)} {obj}، {fmt(y)} خارج شد و {fmt(z)} برگشت.',f'از {fmt(x)} {obj}، {fmt(y)} خارج شد و {fmt(z)} دیگر هم خارج شد.')
            en=(f'From {fmt(x)} {obj}s, {fmt(y)} left and {fmt(z)} returned.',f'From {fmt(x)} {obj}s, {fmt(y)} left and another {fmt(z)} left.')
            if variant==3:
                fa=(f'شمار اولیهٔ {obj} {fmt(x)} است. خروج: {fmt(y)}؛ بازگشت: {fmt(z)}. حالا چندتا داریم؟',f'شمار اولیهٔ {obj} {fmt(x)} است. خروج نخست: {fmt(y)}؛ خروج دوم: {fmt(z)}. حالا چندتا داریم؟')
                en=(f'Initial {obj} count: {fmt(x)}. Departures: {fmt(y)}; returns: {fmt(z)}. How many now?',f'Initial {obj} count: {fmt(x)}. First departures: {fmt(y)}; second departures: {fmt(z)}. How many now?')
        else:
            den=rng.choice([3,4,5]);part=rng.randrange(1,den);unit=rng.randrange(2,31);x=den*unit;answers=(part*unit,(den-part)*unit)
            fa=(f'{fmt(part)} از هر {fmt(den)} {obj} سالم است؛ کل {fmt(x)} است. تعداد سالم؟',f'{fmt(part)} از هر {fmt(den)} {obj} سالم است؛ کل {fmt(x)} است. تعداد ناسالم؟')
            en=(f'{fmt(part)} out of every {fmt(den)} {obj}s are sound; total {fmt(x)}. How many sound?',f'{fmt(part)} out of every {fmt(den)} {obj}s are sound; total {fmt(x)}. How many defective?')
            if variant==3:
                fa=(f'کل {obj}ها {fmt(x)} است. کسر سالم {fmt(part)}/{fmt(den)} است. شمار سالم را بگو.',f'کل {obj}ها {fmt(x)} است. کسر سالم {fmt(part)}/{fmt(den)} است. شمار ناسالم را بگو.')
                en=(f'There are {fmt(x)} {obj}s. The sound fraction is {fmt(part)}/{fmt(den)}. Give the sound count.',f'There are {fmt(x)} {obj}s. The sound fraction is {fmt(part)}/{fmt(den)}. Give the defective count.')
        if answers[0]==answers[1]:return None
        p,n=[f'پاسخ {obj} برابر {fmt(z)} است.' if lang=='fa' else f'The answer for the {obj} is {fmt(z)}.' for z in answers]
    elif family==4:
        # Equality flips truth between inclusive and strict bounds.
        fa=(f'اندازهٔ {obj} {fmt(x)} است. حد مجاز حداقل {fmt(x)} است.',f'اندازهٔ {obj} {fmt(x)} است. حد مجاز بیشتر از {fmt(x)} است.')
        en=(f'The {obj} measures {fmt(x)}. The limit requires at least {fmt(x)}.',f'The {obj} measures {fmt(x)}. The limit requires more than {fmt(x)}.')
        if variant==3:
            fa=(f'{obj} دقیقاً روی مرز {fmt(x)} قرار دارد؛ برابری با مرز پذیرفته است.',f'{obj} دقیقاً روی مرز {fmt(x)} قرار دارد؛ برابری با مرز پذیرفته نیست.')
            en=(f'The {obj} is exactly on the boundary {fmt(x)}; equality with the boundary is accepted.',f'The {obj} is exactly on the boundary {fmt(x)}; equality with the boundary is not accepted.')
        p,n=(f'{obj} شرط را دارد.',f'{obj} شرط را ندارد.') if lang=='fa' else (f'The {obj} meets the requirement.',f'The {obj} fails the requirement.')
    elif family==5:
        x,y=sorted(rng.sample(range(-25,25),2));delta=y-x
        fa=(f'دمای {obj} از {fmt(x)} به {fmt(y)} رسید.',f'دمای {obj} از {fmt(y)} به {fmt(x)} رسید.')
        en=(f'The {obj} temperature went from {fmt(x)} to {fmt(y)}.',f'The {obj} temperature went from {fmt(y)} to {fmt(x)}.')
        if variant==3:
            fa=(f'دماسنج {obj} قبلاً {fmt(x)} را نشان می‌داد و حالا {fmt(y)} را نشان می‌دهد.',f'دماسنج {obj} قبلاً {fmt(y)} را نشان می‌داد و حالا {fmt(x)} را نشان می‌دهد.')
            en=(f'The {obj} thermometer used to read {fmt(x)} and now reads {fmt(y)}.',f'The {obj} thermometer used to read {fmt(y)} and now reads {fmt(x)}.')
        p,n=(f'{obj} {fmt(delta)} درجه گرم‌تر شد.',f'{obj} {fmt(delta)} درجه سردتر شد.') if lang=='fa' else (f'The {obj} warmed by {fmt(delta)} degrees.',f'The {obj} cooled by {fmt(delta)} degrees.')
    elif family==6:
        x=rng.choice([10,20,25,40,50]);y=rng.choice([v for v in range(x+5,96,5)]);delta=y-x;relative=delta*100//x
        fa=(f'درصد {obj} از {fmt(x)} به {fmt(y)} رسید. افزایش با واحد درصد؟',f'درصد {obj} از {fmt(x)} به {fmt(y)} رسید. افزایش نسبی به درصد؟')
        en=(f'The {obj} percentage went from {fmt(x)} to {fmt(y)}. Increase in percentage points?',f'The {obj} percentage went from {fmt(x)} to {fmt(y)}. Relative percent increase?')
        if variant==3:
            fa=(f'سهم {obj} قبلاً {fmt(x)} درصد و اکنون {fmt(y)} درصد است؛ تفاضل مستقیم را بگو.',f'سهم {obj} قبلاً {fmt(x)} درصد و اکنون {fmt(y)} درصد است؛ رشد نسبی نسبت به مقدار قبلی را بگو.')
            en=(f'The {obj} share was {fmt(x)} percent and is now {fmt(y)} percent; give the direct difference.',f'The {obj} share was {fmt(x)} percent and is now {fmt(y)} percent; give the relative growth over the old value.')
        p,n=(f'مقدار افزایش {obj} برابر {fmt(delta)} است.',f'مقدار افزایش {obj} برابر {fmt(relative)} است.') if lang=='fa' else (f'The {obj} increase is {fmt(delta)}.',f'The {obj} increase is {fmt(relative)}.')
        if delta*100%x:return None
    else:
        h=rng.randrange(6,20);m=rng.randrange(30,60);dur=rng.randrange(10,100);end=h*60+m+dur
        # Ending forty minutes later changes the correct answer by exactly forty.
        times=[f'{(end+d)//60}:{(end+d)%60:02d}' for d in [0,40]];start=f'{h}:{m:02d}'
        fa=tuple(f'آماده‌سازی {obj} از {start} تا {t} طول کشید. مدت چند دقیقه بود؟' for t in times)
        en=tuple(f'Preparing the {obj} ran from {start} until {t}. How many minutes?' for t in times)
        if variant==3:
            fa=tuple(f'زمان شروع کار روی {obj}: {start}؛ پایان: {t}. فاصلهٔ زمانی؟' for t in times)
            en=tuple(f'Start time for work on the {obj}: {start}; finish: {t}. Elapsed time?' for t in times)
        p,n=(f'زمان {obj} {fmt(dur)} دقیقه بود.',f'زمان {obj} {fmt(dur+40)} دقیقه بود.') if lang=='fa' else (f'The {obj} took {fmt(dur)} minutes.',f'The {obj} took {fmt(dur+40)} minutes.')
    q0,q1=fa if lang=='fa' else en
    return q0,q1,p,n


def synthetic(split, count=2400):
    rng=random.Random(401 if split=='train' else 402);rows=[]
    objects=[('کمد','cabinet'),('پرده','curtain'),('قاب','frame'),('بشقاب','plate'),('نرده','railing'),('کاشی','tile'),('قفسه','shelf'),('سبد','basket'),('میز','table'),('تخته','board'),('ساعت','clock'),('پنجره','window')]
    if split=='val':objects=[('دریچه','hatch'),('نیمکت','bench'),('کشو','drawer'),('آینه','mirror')]
    names=[('رویا','Ruth'),('بهرام','Brian'),('سمیرا','Sally'),('فرهاد','Frank')]
    if split=='val':names=[('نیلوفر','Nina'),('کامران','Colin')]
    for lang in ['fa','en']:
        li=int(lang=='en')
        for i in range(count):
            obj=rng.choice(objects)[li];a=rng.choice(names)[li];family=i%8;variant=3 if split=='val' else rng.randrange(3)
            for cat in ['negation','numbers']:
                pair=negation(obj,a,lang,family,variant) if cat=='negation' else numeric(rng,obj,lang,family,variant)
                if pair is None:continue
                q0,q1,p,n=pair
                if normalize(p)==normalize(n):continue
                key=f'{lang}-{cat}-{i}'
                for flip,(q,pos,neg) in enumerate([(q0,p,n),(q1,n,p)]):
                    rows.append(dict(anchor=q,positive=pos,negative=neg,lang=lang,category=cat,topic=cat+'-'+obj,template=family,wording=variant,pair=key,flip=flip))
    # Deduplicate the entire counterfactual pair, never just one side.
    unique={}
    for i in range(0,len(rows),2):
        pair=rows[i:i+2];key=tuple(r[k] for r in pair for k in ['anchor','positive','negative']);unique[key]=pair
    return [r for pair in unique.values() for r in pair]


def prepare():
    if OUT.exists():raise FileExistsError('Preserve round-four data')
    manifest=json.loads((ROUND3/'manifest.json').read_text());tk=load_tokenizer(manifest['tokenizer'])
    held=json.loads(CHALLENGE.read_text());forbidden={normalize(r['text']) for r in held['corpus']+held['queries']}
    rows={}
    for split in ['val','train']:
        candidates=synthetic(split,320 if split=='val' else 3200);rows[split]=[]
        for i in range(0,len(candidates),2):
            pair=candidates[i:i+2]
            if all(normalize(r[k]) not in forbidden for r in pair for k in ['anchor','positive','negative']):rows[split]+=pair
        if split=='val':forbidden.update(normalize(r[k]) for r in rows[split] for k in ['anchor','positive','negative'])
    topics={t:i for i,t in enumerate(sorted({r['topic'] for v in rows.values() for r in v}))}
    docs={t:i for i,t in enumerate(sorted({normalize(r[k]) for v in rows.values() for r in v for k in ['positive','negative']}))}
    data={}
    for split,samples in rows.items():
        batch=encode_rows(samples,['anchor','positive','negative'],tk)
        batch['group']=torch.tensor([(0 if r['lang']=='fa' else 3)+['negation','numbers'].index(r['category']) for r in samples])
        batch['topic']=torch.tensor([topics[r['topic']] for r in samples]);batch['doc_ids']=torch.tensor([[docs[normalize(r[k])] for k in ['positive','negative']] for r in samples]);data[split]=batch
        assert len(samples)%2==0 and all(samples[i]['pair']==samples[i+1]['pair'] for i in range(0,len(samples),2))
    OUT.mkdir(parents=True);(OUT/'texts.json').write_text(json.dumps(rows,ensure_ascii=False)+'\n')
    manifest.update(round=4,challenge_sha256=hashlib.sha256(CHALLENGE.read_bytes()).hexdigest(),texts_sha256=hashlib.sha256((OUT/'texts.json').read_bytes()).hexdigest(),counts={k:len(v) for k,v in rows.items()},validation='Held-out wording variant 3 and held-out nouns/names; semantic operations remain shared, not an independent benchmark')
    torch.save(data,OUT/'data.pt');(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['counts'],flush=True)

if __name__=='__main__':prepare()
