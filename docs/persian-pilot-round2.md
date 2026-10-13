# دور دوم پایلوت فارسی/انگلیسی

این دور، ادامهٔ checkpoint دورهٔ دهم پایلوت اول با همان mini و توکنایزر ۳۲هزارتایی است. هیچ alias یا قانون پردازش پرسش در مدل یا ارزیابی افزوده نمی‌شود.

## طرح ثبت‌شده پیش از آموزش

- جدول موقعیت قابل‌آموزش `128 × 256` از نوع fp32: ۳۲٬۷۶۸ پارامتر و ۱۳۱٬۰۷۲ بایت اضافه؛ ابعاد بقیهٔ مدل ثابت.
- قالب باینری v2 پس از جدول embedding توکن، جدول موقعیت fp32 را ذخیره می‌کند. موتور جدید v1 بدون موقعیت و v2 با موقعیت را می‌خواند؛ موتور قدیمی v2 را به دلیل نسخهٔ ناشناخته رد می‌کند.
- بازپخش ۳۰هزار جملهٔ فارسی و ۳۰هزار جملهٔ انگلیسی از بخش آموزشی دور اول، به‌علاوهٔ ۲۰هزار جملهٔ AllNLI با بردار همان معلم چندزبانه.
- AllNLI برای سه‌تایی‌های درست/متضاد؛ نمونه‌های تألیفی متقارن برای نفی، مقایسهٔ اعداد، تفریق و نقش‌ها. تقارن یعنی متن منفی همیشه گزینهٔ غلط نیست.
- فقط بخش آموزشی STS Benchmark برای امتیاز انسانی انگلیسی؛ دادهٔ `MCINext/synthetic-persian-sts` برای شباهت فارسی. برچسب‌های مجموعهٔ فارسی تولیدشده‌اند و انسانی محسوب نمی‌شوند. FarSick فقط بخش آزمون دارد و هیچ نمونه‌ای از آن وارد آموزش نمی‌شود.
- ۸ دوره، seed=75، AdamW با lr=0.00005، QAT از ابتدا، batch بازپخش ۶۴، batch سه‌تایی ۲۴ و batch شباهت ۳۲ با تعداد برابر فارسی/انگلیسی.
- تابع هزینه: cosine distillation + 0.1 relational + 0.6 explicit-triplet soft margin + 0.03 CoSENT. هیچ منفی تصادفی از تمام اعضای batch تحمیل نمی‌شود.
- انتخاب checkpoint فقط با بیشینهٔ `validation English Spearman + validation Persian Spearman + 0.5 × validation triplet accuracy` انجام می‌شود. امتیازهای آزمون در انتخاب دخالت ندارند.
- آزمون تازه `eval/quality/corpus/bilingual-hard-round2.json`: ۱۸ پرسش در هر زبان، شش مورد در هر دستهٔ نفی/اعداد/نقش. متن‌ها و پاسخ‌ها قبل از آموزش ثابت می‌شوند. نتیجهٔ دوگزینه‌ای با حاشیهٔ بیش از 1e-6 نیز گزارش می‌شود.
- FarSick و STS Benchmark آزمون‌های رگرسیونِ از قبل دیده‌شده در گزارش دور اول هستند؛ فقط چالش دور دوم از نظر طراحی تازه است. سابقهٔ دیدن داده‌ها در پیش‌آموزش معلم و مدل‌های قبلی معلوم نیست.
- معیار پذیرش: انگلیسی STS حداقل ۹۷٪ mini اصلی، فارسی STS کمتر از دور اول نشود، هر سه دستهٔ چالش تازه در فارسی نسبت به دور اول بهتر شوند، gzip حداکثر ۱۰٪ بزرگ‌تر از دور اول. این اهداف پیش‌بینی موفقیت نیستند.

## اجرا

```sh
.venv/bin/python training/distill/prep/round2.py
.venv/bin/python training/distill/round2.py --smoke
.venv/bin/python training/distill/round2.py --stop-after-epochs 4
```

داده و لاگ‌ها در `training/distill/cache/mini-fa-en-round2` و `dist/pilot-fa-en-v2` ذخیره می‌شوند. اجرای واقعی و ارزیابی WASM کامل شده است؛ [نتایج، شکست‌ها و اندازهٔ نهایی](persian-pilot-round2-results.md) را ببینید.

منابع: [AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli)، [STS Benchmark](https://huggingface.co/datasets/mteb/stsbenchmark-sts)، [Persian synthetic STS](https://huggingface.co/datasets/MCINext/synthetic-persian-sts).

## اصلاح روش بر اساس اعتبارسنجی، پیش از بازکردن آزمون تازه

پس از سه دوره، نفیِ اعتبارسنجی خوب شد اما نقش‌های فارسی در نسخهٔ int4 حدود ۴۹٪ و اعداد حدود ۶۵٪ بودند. بنابراین اجرای نخست پس از چهار دورهٔ کامل متوقف می‌شود و از checkpoint منتخب همان مرحله، چهار دورهٔ متمرکز ادامه می‌یابد. نسخهٔ اسکریپت و طرح نخست در `dist/pilot-fa-en-v2` حفظ شده‌اند؛ این اصلاح یک تصمیم تطبیقی بر اساس اعتبارسنجی است و نباید «تماماً از ابتدا ثابت» توصیف شود.

در مرحلهٔ متمرکز، از هر دستهٔ نفی/اعداد/نقش و هر زبان چهار سه‌تایی در batch می‌آید، به‌علاوهٔ هشت AllNLI. وزن هزینهٔ سه‌تایی از ۰٫۶ به ۱٫۵ و نرخ یادگیری موقعیت به ده برابر نرخ پایه می‌رسد. معیار انتخاب در این مرحله، مجموع Spearman انگلیسی/فارسی به‌علاوهٔ نصف میانگینِ شش دستهٔ هدف است؛ میانگین کلی AllNLI نمی‌تواند ضعف نقش‌ها را پنهان کند. بازپخش و هزینهٔ STS تغییری نمی‌کنند. آزمون نهایی همچنان دست‌نخورده است.

```sh
.venv/bin/python training/distill/round2.py --curriculum --smoke
.venv/bin/python training/distill/round2.py --curriculum
```

گزینهٔ `--stop-after-epochs 4` پایان مرحلهٔ اول را بازتولید می‌کند و افق هشت‌دوره‌ای scheduler آن مرحله را حفظ می‌کند؛ در اجرای اولیه توقف کنترل‌شده پس از ذخیرهٔ دورهٔ چهارم انجام شد.

## بسته‌بندی و مقایسهٔ خروجی منتخب

```sh
.venv/bin/python training/pack/pack.py \
  --ckpt training/distill/runs/mini-fa-en-round2-curriculum/best.pt \
  --embedding-format int4 --output training/pack/out/pilot-fa-en-v2/model.bin
BIN="$PWD/training/pack/out/pilot-fa-en-v2/model.bin" \
TOKENIZER="$PWD/training/pack/out/pilot-fa-en-v2/model.tokenizer.json" \
PKG=packages/mini TARGET=web WEB_OUT=dist/pilot-fa-en-v2 bash scripts/build-engine.sh
.venv/bin/python eval/quality/bilingual.py \
  --evaluation training/distill/cache/mini-fa-en-32k/evaluation.json \
  --candidate dist/pilot-fa-en-v2/d256/tern_engine.js \
  --candidate-model training/pack/out/pilot-fa-en-v2/model.bin \
  --retrieval eval/quality/corpus/bilingual-hard-round2.json \
  --output dist/pilot-fa-en-v2/comparison.json
.venv/bin/python eval/quality/sts-heldout.py \
  --dataset dist/pilot-fa-en/external-sts.json \
  --model candidate=dist/pilot-fa-en-v2/d256/tern_engine.js \
  --output dist/pilot-fa-en-v2/sts-candidate.json
```

اسکریپت آماده‌سازی و اجرای کامل، خروجی موجود را بازنویسی نمی‌کنند. برای اجرای مستقل، از checkout/مسیر خروجی جدا استفاده کنید و آزمون مشاهده‌شدهٔ این دور را به‌عنوان آزمون تازه معرفی نکنید.
