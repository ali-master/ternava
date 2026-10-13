# نتیجهٔ دور چهارم پایلوت — ۲۰۲۶-۱۰-۱۲

مقایسه با خروجی WASM واقعی و cosine خالص انجام شد؛ بدون aliases، قوانین پرسش، ماشین‌حساب، reranker یا API استنتاج. معماری و توکنایزر دور سوم ثابت‌اند.

**تمام اهداف برآورده نشدند؛ مدل جایگزین خودکار دور سوم یا مثال‌ها نمی‌شود.**

شش دورهٔ آموزش، جمع زمان دوره‌ها 19.2 دقیقه. checkpoint دورهٔ 1 با validation انتخاب شد؛ پذیرش شرایط validation: False. چالش تازه پس از انتخاب مصرف شد و برای تنظیم مجدد استفاده نشد.

نتیجهٔ کلیدی: نفی فارسی از ۹/۱۲ به ۸/۱۲، عدد فارسی از ۴/۱۲ به ۵/۱۲، و STS انگلیسی از ۰٫۷۶۹۹ به ۰٫۷۶۳۹ شد. رشد عدد فارسی فقط مربوط به جهت تغییر دمای منفی است؛ شاهد حل محاسبات عمومی نیست. رتبه‌بندی چالش تازهٔ انگلیسی بهتر شد، اما رتبه‌بندی رگرسیون دور دوم افت محسوس داشت. بنابراین ادعای رفع کامل ضعف تعمیم یا حفظ تمام کیفیت قبلی صحیح نیست.

## علت و اصلاح

- هر هشت قالب نفی دور سوم بین train/validation مشترک بودند و هر زبان فقط ۲۷۶ نمونهٔ نفی یکتا داشت. baseline دور سوم روی نگارش جداگانهٔ دور چهارم در نفی به ۶۱٫۱٪ در هر دو زبان افت کرد؛ امتیاز همان مدل روی validation قدیمی ۱۰۰٪ بود. این شکاف به ارزیابی وابسته به قالب اشاره دارد، نه حل نفی آزاد.
- در دادهٔ درصد قبلی، بعضی منفی‌ها جهت افزایش/کاهش را می‌سنجیدند و نه مقدار محاسبه. نمونه‌های جدید دو پرسش خلاف‌واقع با همان دو جواب دارند؛ هر دو جهت با هم در batch دیده می‌شوند. پاسخ‌های عددی برابر حتی با نگارش متفاوت حذف می‌شوند.
- حاشیهٔ تضاد هزینهٔ مستقل دارد. پذیرش هر چهار گروه هدف مستقل است؛ رشد NDCG اجازهٔ پوشاندن افت عدد یا نفی را ندارد. برای حفظ رتبه‌بندی و انگلیسی، بازپخش و حفظ بردارهای دور سوم و STS ادامه یافت.
- این اجرا چند تغییر را هم‌زمان اعمال می‌کند و آزمایش ablation جداگانه ندارد؛ سهم علّی دقیق هر تغییر از این نتیجه قابل استخراج نیست.

## چالش تازه

۳۲ پرسش در برابر تمام ۶۴ سند در هر زبان: ۱۲ نفی، ۱۲ عدد، ۴ نقش و ۴ بازیابی. ممیزی ۴۹۷٬۴۵۰ متنِ آموزش/اعتبارسنجی هیچ تطابق دقیق یا هشت‌کلمه‌ای پیدا نکرد. این آزمون تألیفی کوچک است؛ نبودن در پیش‌آموزش یا استقلال معنایی از قالب‌ها تضمین نشده. پس از این اجرا، این مجموعه نیز رگرسیون است.

| مدل | فارسی NDCG@10 | انگلیسی NDCG@10 | فارسی top1 سخت‌گیرانه | انگلیسی top1 سخت‌گیرانه |
|---|---:|---:|---:|---:|
| mini اصلی | 0.5975 | 0.8344 | 12/32 | 15/32 |
| base اصلی | 0.4420 | 0.8438 | 6/32 | 17/32 |
| دور اول | 0.7737 | 0.8138 | 15/32 | 18/32 |
| دور دوم | 0.7022 | 0.7526 | 14/32 | 17/32 |
| دور سوم | 0.7633 | 0.8452 | 17/32 | 22/32 |
| دور چهارم | 0.7573 | 0.8773 | 16/32 | 24/32 |

top1 سخت‌گیرانه یعنی رتبهٔ اول درست و حاشیهٔ آن نسبت به متضاد بیشتر از 1e-6 باشد. مساوی موفقیت شمرده نمی‌شود. تشخیص تضاد در جدول زیر مستقل از رتبه‌بندی کل مجموعه است.

| زبان / گروه | mini | base | دور اول | دور دوم | دور سوم | دور چهارم |
|---|---:|---:|---:|---:|---:|---:|
| fa / negation | 8/12 (مساوی 0) | 8/12 (مساوی 0) | 7/12 (مساوی 0) | 7/12 (مساوی 0) | 9/12 (مساوی 0) | 8/12 (مساوی 0) |
| fa / numbers | 7/12 (مساوی 3) | 5/12 (مساوی 3) | 6/12 (مساوی 0) | 4/12 (مساوی 0) | 4/12 (مساوی 0) | 5/12 (مساوی 0) |
| fa / role-order | 0/4 (مساوی 4) | 0/4 (مساوی 4) | 0/4 (مساوی 4) | 3/4 (مساوی 0) | 2/4 (مساوی 0) | 2/4 (مساوی 0) |
| fa / retrieval | 4/4 (مساوی 0) | 3/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) |
| en / negation | 9/12 (مساوی 0) | 9/12 (مساوی 0) | 9/12 (مساوی 0) | 7/12 (مساوی 0) | 8/12 (مساوی 0) | 9/12 (مساوی 0) |
| en / numbers | 3/12 (مساوی 0) | 4/12 (مساوی 0) | 7/12 (مساوی 0) | 6/12 (مساوی 0) | 8/12 (مساوی 0) | 8/12 (مساوی 0) |
| en / role-order | 0/4 (مساوی 4) | 0/4 (مساوی 4) | 0/4 (مساوی 4) | 2/4 (مساوی 0) | 3/4 (مساوی 0) | 3/4 (مساوی 0) |
| en / retrieval | 3/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) | 4/4 (مساوی 0) |

بازهٔ bootstrap توصیفی ۹۵٪ با ۱۰هزار بازنمونه‌گیری جفت‌شده برای تغییر NDCG نسبت به دور سوم:

- fa: -0.0061، بازه [-0.0505, +0.0363].
- en: +0.0321، بازه [-0.0115, +0.0807].

نمونه کوچک است و پرسش‌ها وابستگی موضوعی دارند؛ این بازه‌ها اثبات برتری عمومی نیستند.

## رگرسیون‌های قبلی

| مدل | چالش ۲ فارسی | چالش ۲ انگلیسی | چالش ۳ فارسی | چالش ۳ انگلیسی |
|---|---:|---:|---:|---:|
| mini اصلی | 0.6187 | 0.7633 | 0.4767 | 0.7386 |
| base اصلی | 0.5207 | 0.7877 | 0.4177 | 0.7540 |
| دور اول | 0.7724 | 0.7039 | 0.7022 | 0.7347 |
| دور دوم | 0.6931 | 0.6717 | 0.7062 | 0.7431 |
| دور سوم | 0.8565 | 0.8453 | 0.7375 | 0.7825 |
| دور چهارم | 0.7804 | 0.7356 | 0.7981 | 0.8098 |

| مدل | STS فارسی، ۸۵۶۶ جفت | STS انگلیسی، ۱۳۶۵ جفت |
|---|---:|---:|
| mini اصلی | 0.5349 | 0.7534 |
| base اصلی | 0.5036 | 0.7924 |
| دور اول | 0.6303 | 0.7106 |
| دور دوم | 0.6362 | 0.7722 |
| دور سوم | 0.6381 | 0.7699 |
| دور چهارم | 0.6382 | 0.7639 |

STSها آزمون ثابت FarSick و STS Benchmark و معیار Spearman هستند؛ تازه و ندیده محسوب نمی‌شوند. نتایج مدل‌های قدیمی فقط با SHA یکسان WASM و dataset بازاستفاده شدند؛ دور چهارم روی WASM تازه اجرا شد.

## انتخاب checkpoint و تعمیم توسعه

| epoch | پذیرش | نفی فارسی | عدد فارسی | نفی انگلیسی | عدد انگلیسی | STS انگلیسی | رتبه‌بندی قدیم فارسی | رتبه‌بندی قدیم انگلیسی |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | True | 0.6111 | 0.5556 | 0.6111 | 0.4444 | 0.8172 | 0.8487 | 0.8639 |
| 1 | False | 0.6944 | 0.5000 | 0.6667 | 0.5278 | 0.8151 | 0.8658 | 0.8585 |
| 2 | False | 0.6944 | 0.5278 | 0.6389 | 0.5000 | 0.8149 | 0.8821 | 0.8495 |
| 3 | False | 0.6944 | 0.5000 | 0.6389 | 0.5556 | 0.8142 | 0.8594 | 0.8548 |
| 4 | False | 0.6944 | 0.5000 | 0.5833 | 0.5556 | 0.8139 | 0.8754 | 0.8609 |
| 5 | False | 0.6944 | 0.5000 | 0.5833 | 0.5833 | 0.8164 | 0.8748 | 0.8679 |
| 6 | False | 0.6944 | 0.4722 | 0.6389 | 0.5278 | 0.8159 | 0.8601 | 0.8653 |

validation جدید نگارش و نام‌های جدا دارد، ولی عملیات معنایی مشترک‌اند؛ امتیاز بالا، تعمیم به تمام صورت‌های نفی/عدد را ثابت نمی‌کند. دادهٔ جدید ۱۴٬۴۰۶ سه‌تایی آموزش و ۱٬۳۹۸ اعتبارسنجی است. در هر زبان حدود ۹۸۰ سه‌تایی نفی یکتا و حدود ۶۲۰۰ عددی؛ بقیهٔ داده‌ها و نگهداری از دورهای قبلی بازاستفاده شدند.

## حجم واقعی

| مدل | WASM شامل موتور/توکنایزر/وزن | gzip سطح ۹ |
|---|---:|---:|
| mini اصلی | 7,575,678 | 5,047,123 |
| base اصلی | 10,228,415 | 7,170,702 |
| دور اول | 7,873,832 | 5,225,124 |
| دور دوم | 8,004,767 | 5,347,238 |
| دور سوم | 8,004,767 | 5,347,369 |
| دور چهارم | 8,004,767 | 5,347,458 |

تعداد پارامترها: 9,901,952؛ وزن بسته‌بندی‌شده: 5,165,680 بایت؛ تغییر gzip نسبت به دور سوم: +89 بایت (+0.0017٪).

## شرایط قبولی ثبت‌شده

- `validation_accepted`: **رد**
- `english_sts_retained_99_5_percent`: **رد**
- `persian_sts_retained_99_percent`: **قبول**
- `round3_fa_ndcg_retained_99_percent`: **قبول**
- `round3_en_ndcg_retained_99_percent`: **قبول**
- `round4_fa_ndcg_retained_99_percent`: **قبول**
- `round4_en_ndcg_retained_99_percent`: **قبول**
- `fresh_fa_negation_improved`: **رد**
- `fresh_fa_numbers_improved`: **قبول**
- `fresh_en_negation_retained`: **قبول**
- `fresh_en_numbers_retained`: **قبول**
- `raw_bytes_unchanged`: **قبول**
- `wasm_bytes_unchanged`: **قبول**
- `gzip_growth_at_most_1_percent`: **قبول**

## تمام پرسش‌های تازه و خروجی واقعی

| زبان | پرسش | پاسخ رتبهٔ اول دور چهارم | رتبهٔ درست | حاشیهٔ درست منهای غلط |
|---|---|---|---:|---:|
| fa | هیچ‌کس نگفته تمام باغچه‌ها خشک‌اند؛ فقط این را می‌دانیم که باغچهٔ شمالی هنوز خیس است. | همهٔ باغچه‌ها خشک هستند. | 2 | -0.150972 |
| fa | این‌طور نیست که تعمیر آسانسور بی‌فایده نباشد؛ ارزیابی نهایی آن را کاملاً بی‌فایده دانسته است. | تحویل نان همچنان متوقف است. | 4 | -0.010663 |
| fa | از نبود شاهد برای آلودگی عسل نمی‌توان سالم‌بودنش را نتیجه گرفت؛ هر دو احتمال بازند. | کپسول شرط وزن را برآورده نمی‌کند. | 2 | +0.261781 |
| fa | بازرس ادعای «هیچ چراغی روشن نیست» را رد کرد و چراغ روشن راهرو را نشان داد. | حداقل یک چراغ روشن است. | 1 | +0.107876 |
| fa | ما توقف تحویل نان را لغو نکردیم؛ دستور توقف همچنان برقرار است. | تحویل نان همچنان متوقف است. | 1 | +0.353350 |
| fa | احتمال خرابی حسگر را رد نمی‌کنم، ولی وقوع خرابی هم هنوز ثابت نشده است. | ممکن است حسگر خراب باشد. | 1 | +0.406346 |
| fa | نمایشگاه فقط وقتی بسته می‌ماند که برق قطع باشد؛ برق قطع نیست و طبق همین قاعده نمایشگاه باز است. | تمام چراغ‌ها خاموش‌اند. | 2 | +0.016603 |
| fa | نه گزارش اول دقیق بود و نه گزارش دوم؛ بازرس هر دو را نادرست اعلام کرد. | یکی از دو گزارش دقیق بود. | 2 | -0.130995 |
| fa | لغو نکردن تمرین به معنی برگزارشدنش نبود؛ تمرین اصلاً شروع نشد. | تمرین برگزار نشد. | 1 | +0.448075 |
| fa | این رصدخانه دیگر تصویر زنده نمی‌فرستد، هرچند تا هفتهٔ قبل می‌فرستاد. | ارسال تصویر زنده در رصدخانه متوقف شده است. | 1 | +0.107787 |
| fa | به‌جز آوا که موافقت کرد، هیچ‌کدام از اعضای هیئت با خرید تلسکوپ موافق نبودند. | تمرین برگزار نشد. | 16 | -0.422818 |
| fa | برای تحویل کلید، داشتن رسید هم لازم است و هم کافی؛ نگهبان رسید دارد. | نگهبان مجاز به تحویل گرفتن کلید است. | 1 | +0.219444 |
| fa | از ۷۳ نهال، ۲۸ نهال کاشته شد و ۶ نهال تازه آوردند؛ چند نهال برای کاشت مانده؟ | ۳۹ نهال برای کاشت باقی مانده است. | 2 | -0.093948 |
| fa | قیمت بوم نقاشی ۳۶۰ واحد بود و فروشنده یک‌چهارم آن را کم کرد. | قیمت بوم پس از تخفیف ۳۳۵ واحد است. | 2 | -0.110720 |
| fa | کپسول با وزن ۴٫۲ کیلوگرم باید کمتر از ۴٫۲ کیلوگرم باشد تا پذیرفته شود. | قیمت بوم پس از تخفیف ۲۷۰ واحد است. | 4 | -0.103956 |
| fa | حوضچه از منفی نه درجه به منفی چهار درجه رسید. | حوضچه پنج درجه گرم‌تر شده است. | 1 | +0.029758 |
| fa | سهم انرژی خورشیدی از ۱۶ درصد به ۲۸ درصد رسید؛ افزایش را با واحد درصد بیان کن. | سهم انرژی خورشیدی ۱۲ واحد درصد افزایش یافت. | 1 | +0.086276 |
| fa | خشک‌کردن سفال از ساعت ۱۱:۵۰ تا ۱۲:۳۵ ادامه داشت. | خشک‌کردن سفال هشتادوپنج دقیقه طول کشید. | 2 | -0.057440 |
| fa | دو سوم از ۵۷ بادکنک ترکیده‌اند؛ چند بادکنک سالم مانده است؟ | سی‌وهشت بادکنک سالم مانده است. | 2 | -0.047914 |
| fa | حداقل سابقهٔ لازم ۱۴ ماه است؛ داوطلب دقیقاً چهارده ماه سابقه دارد. | داوطلب شرط سابقه را دارد. | 1 | +0.089164 |
| fa | شمار کبوترها ۴۶ بود؛ ۹ تا رفتند و ۳ تا برگشتند. | اکنون سی‌وچهار کبوتر حضور دارند. | 4 | -0.030717 |
| fa | خطا از ۱۲ درصد به ۹ درصد رسید؛ کاهش با واحد درصد چقدر است؟ | خطا سه واحد درصد کاهش یافته است. | 1 | +0.100067 |
| fa | هر بسته شش پیچ دارد؛ از پنج بسته، هفت پیچ مصرف شده است. | بیست‌وسه پیچ باقی مانده است. | 1 | +0.161358 |
| fa | هزینهٔ ورود ۸۰ بود؛ اول ۲۵ درصد تخفیف خورد، سپس ده واحد کارمزد اضافه شد. | قیمت بوم پس از تخفیف ۳۳۵ واحد است. | 11 | -0.061894 |
| fa | مهسا نقشه را از شایان قرض گرفت؛ قرض‌دهنده چه کسی بود؟ | مهسا نقشه را به شایان قرض داد. | 2 | -0.006617 |
| fa | مربی که پرستار را صدا زد، از اتاق بیرون رفت. | مربی پرستار را صدا زد. | 1 | +0.008578 |
| fa | نامه را ناهید برای کاوه نوشت، نه برعکس. | نویسنده ناهید و گیرنده کاوه بود. | 1 | +0.023793 |
| fa | این عکاس بود که از سوی نوازنده دعوت شد. | عکاس نوازنده را دعوت کرد. | 2 | -0.016260 |
| fa | چطور نوشته‌های یک وب‌سایت را بدون تغییر زمان اجرا برای خوانندهٔ فارسی آماده کنم؟ | ترجمهٔ متن و تنظیم جهت راست‌به‌چپ، وب‌سایت را برای فارسی آماده می‌کند. | 1 | +0.269891 |
| fa | پس از خاموشی برق، می‌خواهم بدانم کدام اطلاعات ذخیره‌شده واقعاً باقی مانده‌اند. | سی‌وهفت پیچ باقی مانده است. | 2 | +0.340717 |
| fa | می‌خواهم فاصلهٔ بین دو مبدأ تقویمی را حساب کنم، نه طول متن تاریخ را. | اختلاف دو تاریخ را پس از تبدیل آن‌ها به یک تقویم محاسبه کنید. | 1 | +0.403266 |
| fa | برای آهنگی که صدای خواننده رویش نیست چه عبارتی جستجو کنم؟ | نسخهٔ بی‌کلام آهنگ، موسیقی بدون صدای خواننده است. | 1 | +0.207127 |
| en | Nobody said all the garden beds were dry; the northern bed is still wet. | All garden beds are dry. | 2 | -0.248725 |
| en | It is not the case that repairing the lift is not pointless; the final assessment calls it entirely pointless. | Repairing the lift serves no purpose. | 1 | +0.083148 |
| en | The absence of evidence that the honey is contaminated does not establish its safety; both possibilities remain open. | The safety of the honey is still unknown. | 1 | +0.255214 |
| en | The inspector rejected the claim that no lights were on and pointed to the lit corridor lamp. | At least one light is on. | 1 | +0.101894 |
| en | We did not revoke the suspension of bread deliveries; the suspension remains in force. | Bread deliveries remain suspended. | 1 | +0.511857 |
| en | I do not rule out a sensor fault, although no fault has been established. | The guard is not entitled to collect the key. | 22 | -0.130207 |
| en | The exhibition stays closed only during a power outage; there is no outage and under this rule it is open. | The exhibition is currently closed. | 10 | -0.352905 |
| en | Neither the first report nor the second was accurate; the inspector declared both incorrect. | Neither of the two reports was accurate. | 1 | +0.272219 |
| en | Not cancelling the rehearsal did not mean it took place; it never began. | The rehearsal did not take place. | 1 | +0.246799 |
| en | This observatory no longer streams live images, although it did until last week. | The observatory has stopped streaming live images. | 1 | +0.103001 |
| en | Apart from Ava, who agreed, none of the panel members supported buying the telescope. | Ava supported buying the telescope. | 1 | +0.032375 |
| en | A receipt is both necessary and sufficient for collecting the key; the guard has a receipt. | The guard is entitled to collect the key. | 1 | +0.362291 |
| en | Of 73 saplings, 28 were planted and 6 new ones arrived. How many remain to be planted? | 39 saplings remain to be planted. | 2 | -0.000921 |
| en | A canvas cost 360 units before the seller reduced it by a quarter. | The discounted canvas costs 270 units. | 1 | +0.017128 |
| en | A cylinder weighing 4.2 kilograms must weigh less than 4.2 kilograms to be accepted. | The cylinder fails the weight requirement. | 1 | +0.104437 |
| en | The pool moved from minus nine degrees to minus four. | The pool became five degrees warmer. | 1 | +0.060412 |
| en | Solar energy rose from 16 percent to 28 percent; express the increase in percentage points. | Solar energy increased by 12 percentage points. | 1 | +0.022698 |
| en | Drying the pottery lasted from 11:50 until 12:35. | Drying the pottery took forty-five minutes. | 1 | +0.013181 |
| en | Two thirds of 57 balloons have burst; how many remain intact? | Thirty-eight balloons remain intact. | 2 | -0.030905 |
| en | The minimum experience is 14 months; the applicant has exactly fourteen months. | The applicant meets the experience requirement. | 1 | +0.107055 |
| en | There were 46 pigeons; 9 left and 3 returned. | Thirty-four pigeons are present now. | 2 | -0.046376 |
| en | The error rate fell from 12 percent to 9 percent. What is the decrease in percentage points? | The error rate fell by three percentage points. | 1 | +0.001869 |
| en | Each packet holds six screws; seven screws were used from five packets. | Thirty-seven screws remain. | 2 | -0.066766 |
| en | Admission cost 80 units; a 25 percent discount was applied, then a ten-unit fee was added. | The payable amount is seventy units. | 1 | +0.005589 |
| en | Maya borrowed the map from Sean; who lent it? | Sean lent the map to Maya. | 1 | +0.011007 |
| en | The coach who called the nurse left the room. | The nurse called the coach. | 2 | -0.006927 |
| en | Nora wrote the letter to Kevin, not the other way around. | Nora was the writer and Kevin the recipient. | 1 | +0.009224 |
| en | It was the photographer who was invited by the musician. | The musician invited the photographer. | 1 | +0.002404 |
| en | How can I adapt website text for Persian readers without changing its runtime? | Translate the text and set right-to-left direction to adapt the website for Persian readers. | 1 | +0.415101 |
| en | After a power cut, I need to find out which saved information actually survived. | Test data recovery after a power failure to verify persistence. | 1 | +0.434818 |
| en | I need the interval between two calendar dates rather than the length of their written strings. | Convert the two dates to a common calendar and compute their difference. | 1 | +0.226767 |
| en | What should I search for if I want a song without the singer vocal track? | An instrumental version contains the music without the singer vocals. | 1 | +0.053447 |

## بازتولید و شواهد

- [طرح ثبت‌شدهٔ دور چهارم](persian-pilot-round4.md).
- لاگ‌های آموزش، ممیزی، parity، STS، تمام جزئیات شش مدل و خلاصهٔ JSON در `dist/pilot-fa-en-v4`.
- خروجی وزن: `training/pack/out/pilot-fa-en-v4/model.bin`؛ موتور مرورگر: `dist/pilot-fa-en-v4/d256/tern_engine.js`.
- اجرای داده یک بار پیش از آموزش به علت پشتیبانی نکردن نگارش عدد ۱۲۰۰ متوقف شد؛ تبدیل اعداد چهاررقمی اصلاح و تست شد. هیچ اجرای آموزشی کنار گذاشته یا بر اساس چالش تازه تنظیم نشد.
- ۲۰ تست Python، تطبیق بسته‌بندی روی ۳۲ ورودی و ۱۳۶ آزمون parity توکنایزر/بردار WASM پاس شدند؛ بیشترین اختلاف مؤلفه ۰٫۰۰۰۳۱۸۵ و کمترین cosine برابر ۰٫۹۹۹۹۹۷۷ بود. کد و دادهٔ اجرا با SHA ثبت‌شده تطبیق داده شدند.
- هیچ انتشار، commit یا تعویض مدل مثال‌ها انجام نشد. آزمون انسانی دامنهٔ مستقل و ablation چندبازویی انجام نشده است.

WASM SHA-256: `a2e343574637453c83f56ee79959dac64487bb3c9cc2ae88a7a1e9bd00447c65`
توکنایزر SHA-256: `dddc3fca56fcd694e8b2268d719b7f73827f8fd73d06aedc2e0304cdfc6f08fe`
چالش SHA-256: `777f43c9566d5fbca89f128058141a59f1068216f725045273be7be6c1cd68d7`

## تشخیص جفت‌های خلاف‌واقع روی توسعه

۱۶ جفت در هر گروه از validation جدید با seed ثابت انتخاب شد؛ چالش نهایی نیست. موفقیت هر جفت مستلزم پاسخ درست به هر دو جهت مخالف است.

| گروه | دور سوم با embedding کامل | دور سوم int4 | دور چهارم int4 |
|---|---:|---:|---:|
| fa_negation | 6/16 | 5/16 | 10/16 |
| fa_numbers | 3/16 | 4/16 | 2/16 |
| en_negation | 5/16 | 5/16 | 6/16 |
| en_numbers | 0/16 | 0/16 | 3/16 |

روی این نمونهٔ محدود، ضعف عددی انگلیسی دور سوم با embedding کامل هم وجود داشت؛ فشرده‌سازی int4 به‌تنهایی توضیحش نمی‌دهد. جفت‌های دادهٔ تازه بعد از tokenization همگی متمایزند. در AllNLI بازپخش‌شدهٔ قبلی دو سه‌تایی با پاسخ درست/غلط کاملاً یکسان شناسایی شد؛ دادهٔ موروثی این اجرا آن‌ها را دارد، اما نمونه‌های تازه چنین برخوردی ندارند. این دو نمونه علت کلی ضعف عددی فارسی/انگلیسی را ثابت نمی‌کنند.
