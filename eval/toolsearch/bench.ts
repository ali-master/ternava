// Compare @ternlight/base and @usex/ternava for tool search, English and Persian.
// Path to a mikrotik-mcp checkout (the 1006-tool catalog under test).
const REPO = process.env.MIKROTIK_MCP;
if (!REPO) throw new Error("set MIKROTIK_MCP=/path/to/mikrotik-mcp");
const { moduleCatalog } = await import(`${REPO}/src/tools/index.ts`);
const { buildToolIndex, searchToolIndex } = await import(`${REPO}/src/core/tool-search.ts`);
const { semanticText, fuseRanks } = await import(`${REPO}/src/core/tool-semantic.ts`);
const base = await import(`${REPO}/node_modules/@ternlight/base`);
const ternava = await import("../../packages/mini/src/index.mjs");

const META = new Set(["find_tools", "describe_tool", "invoke_tool"]);
const tools: any[] = [];
for (const mod of moduleCatalog)
  for (const t of mod.tools)
    if (!META.has(t.name))
      tools.push({
        name: t.name,
        title: t.title,
        description: t.description,
        module: mod.slug,
        group: mod.group,
        params: Object.keys(t.inputSchema ?? {}),
      });
const index = buildToolIndex(tools);
const texts = tools.map((t) => semanticText(t));

// [English request, Persian request, accepted tools]
const CASES: [string, string, string[]][] = [
  ["power down the box", "روتر رو خاموش کن", ["shutdown_system"]],
  ["restart the router", "روتر رو ری‌استارت کن", ["reboot_system"]],
  ["change the upstream name servers to 1.1.1.1", "دی‌ان‌اس روتر رو به 1.1.1.1 تغییر بده", ["set_dns_servers"]],
  ["give my phone access to the wireguard vpn", "یک کاربر جدید به وایرگارد اضافه کن", ["add_wireguard_peer", "onboard_wireguard_user"]],
  ["who is on my network right now", "چه دستگاه‌هایی الان به شبکه وصل هستن", ["list_connected_devices"]],
  ["kick my kid's tablet off the internet", "اینترنت تبلت بچه رو قطع کن", ["block_device"]],
  ["open port 8080 to my home server", "پورت 8080 رو به سرور خونه فوروارد کن", ["forward_port", "create_nat_rule"]],
  ["save a copy of the configuration", "از تنظیمات روتر بکاپ بگیر", ["create_backup", "create_local_backup", "capture_config_snapshot", "create_export"]],
  ["how much cpu and memory is in use", "مصرف پردازنده و رم روتر چقدره", ["get_system_resources", "show_system_dashboard"]],
  ["check if 8.8.8.8 is reachable", "ببین 8.8.8.8 در دسترسه یا نه", ["ping"]],
  ["show me the firewall rules", "قوانین فایروال رو نشونم بده", ["list_filter_rules", "show_firewall_filter"]],
  ["limit download speed of 192.168.1.50 to 5 megabit", "سرعت دانلود 192.168.1.50 رو به ۵ مگابیت محدود کن", ["create_simple_queue", "set_device_limits"]],
  ["set a new password for admin", "رمز کاربر admin رو عوض کن", ["change_password", "update_user"]],
  ["what happened on the router in the last hour", "لاگ‌های روتر رو نشون بده", ["get_logs", "search_logs", "get_system_events"]],
  ["is there a newer routeros version", "نسخه جدید سیستم‌عامل روتر اومده؟", ["check_for_updates", "firmware_check"]],
  ["which wifi networks are around", "شبکه‌های وای‌فای اطراف رو اسکن کن", ["scan_wireless_networks", "scan_wifi_channels"]],
  ["make nas.home resolve to 192.168.1.10", "برای nas.home یک رکورد دی‌ان‌اس به 192.168.1.10 بساز", ["add_dns_static"]],
  ["show the hops to google.com", "مسیر رسیدن به google.com رو نشون بده", ["traceroute", "trace_path"]],
  ["assign 10.0.0.1/24 to ether2", "آی‌پی 10.0.0.1/24 رو روی ether2 بذار", ["add_ip_address"]],
  ["always give my printer the same address", "آی‌پی پرینتر رو ثابت کن", ["make_device_static", "set_device_ip"]],
  ["turn on my desktop pc remotely", "کامپیوترم رو از راه دور روشن کن", ["wake_on_lan"]],
  ["how fast is my internet", "سرعت اینترنت رو تست کن", ["speed_test", "bandwidth_test"]],
  ["make a separate network segment tagged 20", "یک وی‌لن با شماره 20 بساز", ["create_vlan_interface"]],
  ["send traffic for 10.10.0.0/16 through 192.168.1.254", "برای 10.10.0.0/16 یک مسیر از 192.168.1.254 اضافه کن", ["add_route"]],
  ["turn off the ether5 port", "پورت ether5 رو غیرفعال کن", ["disable_interface"]],
];

type Embed = (text: string) => Float32Array;
function unit(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n);
}

function run(label: string, embed: Embed): void {
  const t0 = performance.now();
  const vectors = texts.map((t) => unit(embed(t)));
  const built = performance.now() - t0;
  const near = (q: string): string[] => {
    const v = unit(embed(q));
    return vectors
      .map((d, i) => ({ i, s: d.reduce((a, x, k) => a + x * v[k]!, 0) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, 60)
      .map((h) => tools[h.i].name);
  };
  for (const lang of [0, 1] as const) {
    let sem5 = 0;
    let hyb5 = 0;
    let hyb8 = 0;
    const missed: string[] = [];
    for (const c of CASES) {
      const q = c[lang];
      const ok = (names: string[]) => names.some((n) => c[2].includes(n));
      const semantic = near(q);
      const lexical = searchToolIndex(index, q, 60).map((h: any) => h.name);
      const fused = fuseRanks([lexical, semantic], 8);
      if (ok(semantic.slice(0, 5))) sem5++;
      if (ok(fused.slice(0, 5))) hyb5++;
      if (ok(fused)) hyb8++;
      else missed.push(c[2][0]!);
    }
    console.log(
      `${label} ${lang ? "FA" : "EN"}: semantic top5 ${sem5}/25 | fused top5 ${hyb5}/25 | fused top8 ${hyb8}/25 | missed: ${missed.join(",")}`,
    );
  }
  console.log(`${label}: embedded ${texts.length} tools in ${(built / 1000).toFixed(1)} s`);
}

// Words only, as the baseline.
for (const lang of [0, 1] as const) {
  let n = 0;
  for (const c of CASES)
    if (searchToolIndex(index, c[lang], 5).some((h: any) => c[2].includes(h.name))) n++;
  console.log(`words only ${lang ? "FA" : "EN"}: top5 ${n}/25`);
}
run("ternlight/base", base.embed);
run("ternava", ternava.embed);

// Both models together: words + base + ternava, fused.
{
  const vb = texts.map((t) => unit(base.embed(t)));
  const vt = texts.map((t) => unit(ternava.embed(t)));
  const near = (vs: Float32Array[], e: Embed, q: string) => {
    const v = unit(e(q));
    return vs.map((d, i) => ({ i, s: d.reduce((a, x, k) => a + x * v[k]!, 0) })).sort((a, b) => b.s - a.s).slice(0, 60).map((h) => tools[h.i].name);
  };
  for (const lang of [0, 1] as const) {
    let n5 = 0, n8 = 0;
    for (const c of CASES) {
      const q = c[lang];
      const fused = fuseRanks([searchToolIndex(index, q, 60).map((h: any) => h.name), near(vb, base.embed, q), near(vt, ternava.embed, q)], 8);
      if (fused.slice(0, 5).some((n: string) => c[2].includes(n))) n5++;
      if (fused.some((n: string) => c[2].includes(n))) n8++;
    }
    console.log(`both models ${lang ? "FA" : "EN"}: fused top5 ${n5}/25 | top8 ${n8}/25`);
  }
  const sim = (a: string, b: string) => unit(ternava.embed(a)).reduce((s, x, k) => s + x * unit(ternava.embed(b))[k]!, 0).toFixed(2);
  console.log("ternava sim: خاموش کردن روتر ~ shut down the router", sim("خاموش کردن روتر", "shut down the router"), "| ~ list firewall rules", sim("خاموش کردن روتر", "list firewall rules"));
  console.log("ternava sim: قوانین فایروال ~ firewall rules", sim("قوانین فایروال", "firewall rules"), "| ~ reboot the router", sim("قوانین فایروال", "reboot the router"));
}
