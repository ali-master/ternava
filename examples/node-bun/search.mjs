import init, { similar, engineInfo } from '@usex/ternava/runtime';

await init();
console.log(engineInfo());
console.table(similar(process.argv[2] ?? 'خیلی دوستت دارم', [
    'عشق و علاقهٔ فراوان به تو دارم',
    'راهنمای نصب و راه‌اندازی نرم‌افزار',
    'The weather is sunny today',
]));
