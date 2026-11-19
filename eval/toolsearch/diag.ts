// Path to a mikrotik-mcp checkout (the 1006-tool catalog under test).
const REPO = process.env.MIKROTIK_MCP;
if (!REPO) throw new Error("set MIKROTIK_MCP=/path/to/mikrotik-mcp");
const { moduleCatalog } = await import(`${REPO}/src/tools/index.ts`);
const { semanticText } = await import(`${REPO}/src/core/tool-semantic.ts`);
const base = await import(`${REPO}/node_modules/@ternlight/base`);
const tv = await import("../../packages/mini/src/index.mjs");
const tools: any[] = [];
for (const m of moduleCatalog) for (const t of m.tools) tools.push(t);
const texts = tools.map((t) => semanticText(t));
const titles = tools.map((t) => t.title as string);
const dot = (a: Float32Array, b: Float32Array) => a.reduce((s, x, k) => s + x * b[k]!, 0);
type E = (t: string) => Float32Array;

// [EN, FA informal, FA formal, tool]
const C: [string, string, string, string][] = [
  ["power down the box", "روتر رو خاموش کن", "روتر را خاموش کن", "shutdown_system"],
  ["restart the router", "روتر رو ری‌استارت کن", "روتر را دوباره راه‌اندازی کن", "reboot_system"],
  ["change the upstream name servers to 1.1.1.1", "دی‌ان‌اس روتر رو به 1.1.1.1 تغییر بده", "سرور نام دامنه روتر را به 1.1.1.1 تغییر بده", "set_dns_servers"],
  ["who is on my network right now", "چه دستگاه‌هایی الان به شبکه وصل هستن", "چه دستگاه‌هایی اکنون به شبکه متصل هستند", "list_connected_devices"],
  ["save a copy of the configuration", "از تنظیمات روتر بکاپ بگیر", "از تنظیمات روتر نسخه پشتیبان تهیه کن", "create_backup"],
  ["how much cpu and memory is in use", "مصرف پردازنده و رم روتر چقدره", "میزان مصرف پردازنده و حافظه روتر چقدر است", "get_system_resources"],
  ["show me the firewall rules", "قوانین فایروال رو نشونم بده", "قوانین دیوار آتش را نمایش بده", "list_filter_rules"],
  ["set a new password for admin", "رمز کاربر admin رو عوض کن", "رمز عبور کاربر admin را تغییر بده", "change_password"],
  ["show the router logs", "لاگ‌های روتر رو نشون بده", "گزارش رویدادهای روتر را نمایش بده", "get_logs"],
  ["which wifi networks are around", "شبکه‌های وای‌فای اطراف رو اسکن کن", "شبکه‌های بی‌سیم اطراف را جستجو کن", "scan_wireless_networks"],
  ["how fast is my internet", "سرعت اینترنت رو تست کن", "سرعت اینترنت را آزمایش کن", "speed_test"],
  ["turn off the ether5 port", "پورت ether5 رو غیرفعال کن", "درگاه ether5 را غیرفعال کن", "disable_interface"],
];
// --dump <file>: write tool texts + cases for teacher.py (same benchmark, Python models).
if (process.argv[2] === "--dump") {
  await Bun.write(process.argv[3]!, JSON.stringify({ names: tools.map((t) => t.name), texts, titles, cases: C }));
  process.exit(0);
}
function report(label: string, e: E) {
  const docs = texts.map(e);
  const tdocs = titles.map(e);
  // 1. how narrow is the space: mean cosine of unrelated tool pairs
  let s = 0, n = 0;
  for (let i = 0; i < docs.length; i += 7) for (let j = i + 3; j < docs.length; j += 97) { s += dot(docs[i]!, docs[j]!); n++; }
  const rank = (q: string, d: Float32Array[], want: string) => {
    const v = e(q);
    const order = d.map((x, i) => ({ i, s: dot(x, v) })).sort((a, b) => b.s - a.s);
    const r = order.findIndex((h) => tools[h.i].name === want) + 1;
    return { r, top: order[0]!.s, at: order[r - 1]?.s ?? 0, s50: order[49]!.s };
  };
  console.log(`\n== ${label} == mean cosine of unrelated tool texts: ${(s / n).toFixed(2)}`);
  const med = (a: number[]) => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
  for (const [k, name] of [[0, "EN"], [1, "FA informal"], [2, "FA formal"]] as const) {
    const full = C.map((c) => rank(c[k], docs, c[3]));
    const short = C.map((c) => rank(c[k], tdocs, c[3]));
    console.log(`${name}: median rank of the right tool, full text ${med(full.map((x) => x.r))} | title only ${med(short.map((x) => x.r))} | in top5 ${full.filter((x) => x.r <= 5).length}/12 (title only ${short.filter((x) => x.r <= 5).length}/12) | mean top score ${(full.reduce((a, x) => a + x.top, 0) / 12).toFixed(2)}, mean 50th score ${(full.reduce((a, x) => a + x.s50, 0) / 12).toFixed(2)}`);
  }
  // 2. translation retrieval: does a Persian request find its own English twin among the 12?
  const en = C.map((c) => e(c[0]));
  for (const [k, name] of [[1, "FA informal"], [2, "FA formal"]] as const) {
    let hit = 0, own = 0, other = 0;
    C.forEach((c, i) => {
      const v = e(c[k]);
      const sims = en.map((x) => dot(x, v));
      if (sims.indexOf(Math.max(...sims)) === i) hit++;
      own += sims[i]!;
      other += (sims.reduce((a, x) => a + x, 0) - sims[i]!) / 11;
    });
    console.log(`${name} -> its English twin: top1 ${hit}/12 | mean sim own ${(own / 12).toFixed(2)} vs others ${(other / 12).toFixed(2)}`);
  }
  // 3. technical loanwords
  const W: [string, string][] = [["روتر", "router"], ["فایروال", "firewall"], ["دیوار آتش", "firewall"], ["وایرگارد", "wireguard"], ["دی‌ان‌اس", "dns"], ["پورت", "port"], ["بکاپ", "backup"], ["پشتیبان", "backup"], ["وی‌لن", "vlan"], ["وای‌فای", "wifi"], ["لاگ", "log"], ["آی‌پی", "ip address"], ["رمز عبور", "password"], ["خاموش کردن", "shut down"]];
  const ev = W.map((w) => e(w[1]));
  console.log("word pairs (sim to own English | mean sim to the other English words): " + W.map((w, i) => { const v = e(w[0]); const o = ev.reduce((a, x, j) => a + (j === i ? 0 : dot(x, v)), 0) / (W.length - 1); return `${w[1]}<-${w[0]} ${dot(v, ev[i]!).toFixed(2)}|${o.toFixed(2)}`; }).join("; "));
}
report("ternava", tv.embed);
report("ternlight/base", base.embed);
// 4. normalization details
const sim = (a: string, b: string) => dot(tv.embed(a), tv.embed(b)).toFixed(2);
console.log("\nternava details: ZWNJ vs space 'دی‌ان‌اس'~'دی ان اس'", sim("دی‌ان‌اس", "دی ان اس"), "| joined 'دیاناس'", sim("دی‌ان‌اس", "دیاناس"), "| Persian digits '۵ مگابیت'~'5 مگابیت'", sim("۵ مگابیت", "5 مگابیت"), "| word order 'روتر را خاموش کن'~'خاموش کن را روتر'", sim("روتر را خاموش کن", "خاموش کن را روتر"), "| 'خاموش کن'~'روشن کن'", sim("روتر را خاموش کن", "روتر را روشن کن"), "| en 'enable interface'~'disable interface'", sim("enable interface", "disable interface"));
