"""Latent arithmetic/truth labels with independent surface forms and hard negatives."""
import json
import random

import torch

from diagnose_reasoning import CURRENT, OUT
from prep.native import ROOT, digest
from prep.round2 import encode_rows, normalize
from prep.round4 import render_number
from tokenization import load_tokenizer

CACHE = ROOT / 'training/distill/cache/mini-fa-en-compositional'
CHALLENGE = ROOT / 'eval/quality/corpus/persian-compositional-hard.json'
KEYS = ['query', 'paraphrase', 'positive', 'negative', 'near1', 'near2']
NOUNS = ['باتری','بسته','جعبه','مهره','دکمه','سکه','پاکت','کارت','فایل','رکورد','قطعه','چراغ','حسگر','صندلی','صفحه','تراشه','کتاب','کلید','لیوان','بلیت']
PREDICATES = [('مجاز است','مجاز نیست'),('سالم است','سالم نیست'),('فعال است','فعال نیست'),('موجود است','موجود نیست'),('تایید شده است','تایید نشده است'),('قابل استفاده است','قابل استفاده نیست'),('ارسال شده است','ارسال نشده است'),('حذف شده است','حذف نشده است')]


def answer(obj, value, style):
    return f'مقدار نهایی برای {obj} برابر {render_number(value, "fa", style)} است.'


def numeric_case(rng, index, split):
    obj = rng.choice(NOUNS if split == 'train' else ['قرقره','گیره','مخزن','بشکه'])
    x = rng.randrange(4, 320 if split == 'train' else 480); y = rng.randrange(2, 20); z = rng.randrange(1, 12)
    family = index % 8; sign = rng.choice([-1, 1]); styles = [rng.randrange(3), rng.randrange(3)]
    if family == 0: value = x + sign*y; wrong = x - sign*y
    elif family == 1: value = x-y+sign*z; wrong = x-y-sign*z
    elif family == 2:
        x = max(20, x//20*20); y = rng.choice([10,20,25,50]); value = x*(100+sign*y)//100; wrong=x*(100-sign*y)//100
    elif family == 3:
        y = rng.choice([3,5,7]); z = rng.randrange(1,y); x = max(y,x//y*y)
        value = x//y*(z if sign == 1 else y-z); wrong = x-value
    elif family == 4:
        x = rng.randrange(-30,30); y = rng.randrange(1,20); value = sign*y; wrong = -value
    elif family == 5:
        x = rng.randrange(360,1200); y = rng.randrange(5,160); value=y; wrong=y+60*sign
    elif family == 6:
        x = y + [-1, 0, 1][index % 3]
        value = int(x >= y) if sign == 1 else int(x > y); wrong = 1-value
    else:
        x = rng.randrange(2,20); y = rng.randrange(2,10); z = rng.randrange(1,10)
        value = x*y+sign*z; wrong = x*y-sign*z
    views=[]
    for view,style in enumerate(styles):
        f=lambda n:render_number(n,'fa',style)
        a,b,c=f(x),f(y),f(z); plus=sign==1
        variant=(index+view)%3 if split=='train' else 3
        if family==0:
            verb=['اضافه شد' if plus else 'کم شد','وارد شد' if plus else 'خارج شد','دریافت کردیم' if plus else 'تحویل دادیم','افزوده‌اند' if plus else 'کاسته‌اند'][variant]
            q=[f'از {obj} تعداد {a} داریم؛ {b} {verb}. حالا چندتا؟',f'موجودی {obj}: {a}. تعداد {b} {verb}. مقدار پایان؟',f'ابتدا {a} {obj} داشتیم؛ سپس {b} {verb}. چقدر باقی است؟',f'به موجودی {a}تایی {obj}، {b} {verb}؛ تعداد تازه را می‌خواهم.'][variant]
        elif family==1:
            op=['افزایش' if plus else 'کاهش','برگشت' if plus else 'رفت','خریدیم' if plus else 'فروختیم','اضافه کن' if plus else 'بردار'][variant]
            q=[f'شمار {obj}: {a}؛ کاهش {b}؛ {op} {c}. حاصل؟',f'از {a} {obj}، {b} رفت و {c} {op}. موجودی؟',f'{a} {obj} بود. {b} فروختیم و {c} {op}. تعداد نهایی؟',f'از مجموعهٔ {a}تایی {obj} ابتدا {b} بردار؛ سپس {c} {op}. چند مورد می‌ماند؟'][variant]
        elif family==2:
            op=['افزایش' if plus else 'کاهش','گران‌تر' if plus else 'ارزان‌تر','اضافه‌بها' if plus else 'تخفیف','رشد' if plus else 'افت'][variant]
            q=[f'بهای {obj} {a} است؛ {b} درصد {op}. بهای نهایی؟',f'{obj} از قیمت {a}، {b} درصد {op} شد؛ قیمت تازه؟',f'روی قیمت {a} برای {obj}، {b} درصد {op} اعمال شد. حاصل؟',f'قیمت اولیهٔ {obj} برابر {a} بود. با {op} نسبی {b} درصدی، چه بهایی به دست می‌آید؟'][variant]
        elif family==3:
            target='سالم' if plus else 'ناسالم'
            q=[f'از {a} {obj}، {c} از هر {b} سالم است. تعداد {target}؟',f'کل {obj}ها {a} است؛ سهم سالم {c}/{b}. شمار {target}؟',f'در میان {a} {obj} نسبت سالم {c} به {b} است؛ چند {target} داریم؟',f'مجموع {obj}ها {a} تاست و کسر سالمشان {c}/{b} است؛ گروه {target} چند عضو دارد؟'][variant]
        elif family==4:
            end=f(x+sign*y)
            q=[f'دمای {obj} از {a} به {end} رسید. تغییر علامت‌دار؟',f'دماسنج {obj}: ابتدا {a} و پایان {end}. پایان منهای شروع؟',f'{obj} اول {a} درجه و سپس {end} درجه بود. اختلاف علامت‌دار چیست؟',f'برای {obj} قرائت قبلی {a} و قرائت تازه {end} است؛ تغییر با حفظ علامت را گزارش کن.'][variant]
        elif family==5:
            start=f'{x//60}:{x%60:02d}'; end=f'{(x+y)//60}:{(x+y)%60:02d}'
            q=[f'کار روی {obj} از {start} تا {end} بود. چند دقیقه؟',f'شروع آماده‌سازی {obj} {start} و پایان {end}. مدت به دقیقه؟',f'{obj} را ساعت {start} شروع و ساعت {end} تمام کردیم. چند دقیقه گذشت؟',f'زمان ثبت شروع {obj} برابر {start} و اتمامش {end} در همان روز است؛ فاصلهٔ زمانی به دقیقه؟'][variant]
        elif family==6:
            op='حداقل' if plus else 'بیشتر از'
            q=[f'اندازهٔ {obj} {a} است. شرط: {op} {b}. اگر قبول است ۱ وگرنه ۰.',f'برای {obj} مقدار {a} و آستانهٔ {op} {b} است؛ قبول=۱، رد=۰.',f'آیا {obj} با اندازهٔ {a} شرط {op} {b} را دارد؟ پاسخ عددی: بله ۱، خیر ۰.',f'قاعدهٔ پذیرش {obj}، {op} {b} است. مقدار واقعی {a} را بسنج؛ پذیرش را با ۱ و عدم پذیرش را با ۰ نشان بده.'][variant]
        else:
            op='اضافه' if plus else 'کم'
            q=[f'{a} گروه {b}تایی {obj} داریم؛ {c} مورد {op} شد. کل نهایی؟',f'هر گروه {b} {obj} دارد؛ {a} گروه و سپس {c} {op}. حاصل؟',f'{obj} در {a} دستهٔ {b}تایی است و {c} مورد {op} شده. چندتا؟',f'ابتدا {a} برابر {b} {obj} حساب کن؛ سپس {c} {op} کن. حاصل شمارش؟'][variant]
        if split != 'train' and view == 1:
            q = [
                f'برای {obj} مقدار آغازین {a} و تغییر {"مثبت" if plus else "منفی"} {b} ثبت شده؛ موجودی به چه عددی می‌رسد؟',
                f'از {a} عدد {obj} ابتدا {b} خارج شده و مرحلهٔ بعد {c} مورد {"بازگردانده" if plus else "خارج"} شده است. شمار باقی‌مانده چیست؟',
                f'با مبنای {a} برای قیمت {obj}، تغییر {"مثبت" if plus else "منفی"} {b} درصد چه قیمت تازه‌ای می‌دهد؟',
                f'در مجموعهٔ {a}تایی {obj}، سهم {c}/{b} سالم و مابقی خراب‌اند. تعداد {"سالم‌ها" if plus else "خراب‌ها"} را محاسبه کن.',
                f'مقدار پایانی دمای {obj} برابر {f(x+sign*y)} و مقدار اولیه {a} بوده؛ تفاضل پایانی و اولیه چند است؟',
                f'فاصلهٔ میان ساعت {x//60}:{x%60:02d} تا ساعت {(x+y)//60}:{(x+y)%60:02d} برای کار {obj} را به دقیقه تبدیل کن.',
                f'برای پذیرش {obj} مقدار باید {"کمتر از" if plus else "کوچک‌تر یا مساوی"} {b} نباشد؛ مقدار {a} است. قبول را ۱ و رد را ۰ بنویس.',
                f'شمار {obj} با حاصل ضرب {a} در {b} آغاز می‌شود و با {"افزودن" if plus else "برداشتن"} {c} پایان می‌گیرد؛ نتیجه چیست؟',
            ][family]
        views.append(q)
    candidates=[value,wrong]
    for delta in [-1,1,-2,2,3]:
        if value+delta not in candidates:candidates.append(value+delta)
        if len(candidates)==4:break
    style=rng.randrange(3);docs=[answer(obj,n,style) for n in candidates]
    return dict(zip(KEYS,views+docs),category='numbers',family=family,signature=[['number',obj,n] for n in candidates],value=value,operands=[x,y,z,sign])


def truth_case(rng,index,split):
    obj=rng.choice(NOUNS if split=='train' else ['قرقره','گیره','مخزن','بشکه']);yes,no=rng.choice(PREDICATES)
    base=rng.randrange(2);depth=rng.randrange(1,4);operators=[rng.randrange(2) for _ in range(depth)]
    state=base
    for op in operators:state ^= op
    if index%3==2:state=2
    clauses=[f'{obj} {yes}.',f'{obj} {no}.',f'وضعیت «{obj} {yes}» نامعلوم است.']
    views=[]
    wrappers=[['درست است که «{}»','نادرست است که «{}»'],['این گفته تایید شده: «{}»','این گفته تکذیب شده: «{}»'],['این ادعا حقیقت دارد: «{}»','این ادعا غلط است: «{}»'],['نتیجهٔ بررسی، درستی این گزاره است: «{}»','نتیجهٔ بررسی، نادرستی این گزاره است: «{}»'],['واقعیت داشتن گزارهٔ «{}» تایید می‌شود','واقعیت داشتن گزارهٔ «{}» رد می‌شود']]
    for view in range(2):
        q=clauses[base]
        for j,op in enumerate(operators):q=wrappers[(index+view+j)%3 if split=='train' else 3+view][op].format(q)
        if state==2:q=[f'دربارهٔ درستی یا نادرستی «{q}» هنوز هیچ نتیجه‌ای نداریم.',f'نه این گفته ثابت شده نه نقیضش: «{q}».',f'راستی‌آزماییِ «{q}» پایان نیافته؛ هر دو امکان باقی است.',f'شواهد فعلی دربارهٔ «{q}» برای تعیین درست یا غلط بودن آن کافی نیست.',f'هنوز مشخص نشده که گزارهٔ «{q}» واقعیت دارد یا ندارد.'][index%3 if split=='train' else 3+view]
        if index%2==0:q=f'موضوع دیگر: وضعیت شبکه مشخص نیست. دربارهٔ {obj}: '+q
        views.append(q)
    order=[state]+[v for v in range(3) if v!=state]
    other='موتور' if obj!='موتور' else 'پمپ'
    docs=[clauses[s] for s in order]+[f'{other} {yes}.']
    signatures=[['truth',obj,yes,s] for s in order]+[['truth',other,yes,0]]
    return dict(zip(KEYS,views+docs),category='negation',family=depth,signature=signatures,operators=operators,base=base,state=state)


def generate(split,count):
    rng=random.Random(827 if split=='train' else 828)
    return [maker(rng,i,split) for i in range(count) for maker in [numeric_case,truth_case]]


def prepare():
    if CACHE.exists():raise FileExistsError('Preserve data')
    ckpt=torch.load(CURRENT,map_location='cpu',weights_only=False);tk=load_tokenizer(ckpt['tokenizer'])
    fresh=json.loads(CHALLENGE.read_text());held={normalize(q['text']) for f in fresh['suites'].values() for q in f['queries']}
    texts={s:generate(s,n) for s,n in [('val',160),('train',2400)]}
    val={normalize(r[k]) for r in texts['val'] for k in KEYS[:2]}
    texts['train']=[r for r in texts['train'] if not any(normalize(r[k]) in held|val for k in KEYS[:2])]
    data={}
    for split,rows in texts.items():
        assert all(r['query'] != r['paraphrase'] for r in rows)
        assert not {normalize(r[k]) for r in rows for k in KEYS[:2]}&held
        batch=encode_rows(rows,KEYS,tk)
        assert int(batch['attention_mask'].sum(-1).max())<128,'Truncated supervision'
        assert not (batch['input_ids']==tk.unk_token_id).any()
        for row in batch['input_ids']:
            assert all(not torch.equal(row[2],d) for d in row[3:]),'Indistinguishable negative'
        batch['group']=torch.tensor([r['category']=='numbers' for r in rows])
        labels={s:i for i,s in enumerate(sorted({json.dumps(x,ensure_ascii=False) for r in rows for x in r['signature']}))}
        batch['semantic_ids']=torch.tensor([[labels[json.dumps(x,ensure_ascii=False)] for x in r['signature']] for r in rows])
        data[split]=batch
    CACHE.mkdir(parents=True);torch.save(data,CACHE/'data.pt')
    (CACHE/'texts.json').write_text(json.dumps(texts,ensure_ascii=False)+'\n')
    manifest=dict(source_sha256=digest(CURRENT),challenge_sha256=digest(CHALLENGE),diagnosis_sha256=digest(OUT/'diagnosis.json'),counts={s:len(r) for s,r in texts.items()},files={p.name:digest(p) for p in CACHE.iterdir()},tokenizer_sha256=digest(ckpt['tokenizer']['path']))
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)


if __name__=='__main__':prepare()
