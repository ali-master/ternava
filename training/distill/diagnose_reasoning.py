"""Factorial development probes; never use these consumed cases as final tests."""
import json
import random
import re
from unittest.mock import patch

import torch

from native import embed_all
from prep.native import ROOT, digest
from prep.reasoning import numeric, SOURCE
from prep.round2 import encode_rows
from prep.round4 import negation, render_number
from reasoning import load_model
from tokenization import load_tokenizer

OUT = ROOT / 'dist/pilot-fa-en-compositional'
CURRENT = ROOT / 'training/distill/runs/mini-fa-en-reasoning/english-safe.pt'


class ShiftRange(random.Random):
    def randrange(self, *args, **kwargs):
        value = super().randrange(*args, **kwargs)
        return value + 400 if args == (20, 301) else value


def numeric_probe(state, obj, family, variant, axis):
    rng = ShiftRange() if axis == 'range' else random.Random()
    rng.setstate(state)
    rendered = []

    def fmt(n, lang, style):
        rendered.append(n)
        return render_number(n, lang, 1 if axis == 'digits' else style)

    with patch('prep.reasoning.render_number', fmt):
        pair = numeric(rng, 'قرقره' if axis == 'noun' else obj, 'fa', family,
                       3 if axis == 'wording' else variant)
    values = rendered[-2:]
    docs = list(pair[2:])
    noun = 'قرقره' if axis == 'noun' else obj
    for value in values:
        # Nearest-answer negatives invalidate the old large/small-answer shortcut.
        docs.extend(f'حاصل برای {noun} برابر {render_number(value + d, "fa", 1)} است.' for d in [-1, 1])
    queries = list(pair[:2])
    if axis == 'no_numbers':
        # Query-only intervention: candidate answers stay exactly unchanged.
        number_words = set('صفر یک دو سه چهار پنج شش هفت هشت نه ده یازده دوازده سیزده چهارده پانزده شانزده هفده هجده نوزده بیست سی چهل پنجاه شصت هفتاد هشتاد نود صد دویست سیصد چهارصد پانصد ششصد هفتصد هشتصد نهصد هزار منفی'.split())
        queries = [' '.join('عدد' if w in number_words else re.sub(r'\d+', 'عدد', w) for w in q.split()) for q in queries]
    return dict(category=f'numeric_{family}', axis=axis, queries=queries, docs=docs, values=values)


def probes():
    rng = random.Random(719)
    nouns = ['جعبه', 'مهره', 'دکمه', 'سکه', 'پاکت', 'بسته']
    counts = [0] * 4
    result = []
    for i in range(1400):
        obj = rng.choice(nouns); variant = rng.randrange(3); state = rng.getstate()
        numeric(rng, obj, 'fa', i % 7, variant)
        if i % 7 < 4 and counts[i % 7] < 24:
            counts[i % 7] += 1
            for axis in ['seen', 'range', 'wording', 'noun', 'digits', 'no_numbers']:
                result.append(numeric_probe(state, obj, i % 7, variant, axis))
    for family in range(8):
        for obj in nouns:
            for variant in [0, 1, 2]:
                for axis in ['seen', 'wording', 'noun']:
                    pair = negation('قرقره' if axis == 'noun' else obj, 'مهتاب', 'fa', family, 3 if axis == 'wording' else variant)
                    result.append(dict(category=f'negation_{family}', axis=axis, queries=list(pair[:2]), docs=list(pair[2:])))
    return result


@torch.no_grad()
def score(model, rows, tk, device):
    texts = list(dict.fromkeys(t for r in rows for t in r['queries'] + r['docs']))
    tokens = encode_rows([dict(text=t) for t in texts], ['text'], tk)
    vectors = embed_all(model, tokens, device)[:, 0]
    index = {t: i for i, t in enumerate(texts)}
    buckets = {}
    for r in rows:
        q = vectors[[index[t] for t in r['queries']]]; d = vectors[[index[t] for t in r['docs']]]
        scores = q @ d.T; gap = torch.stack([scores[0, 0] - scores[0, 1], scores[1, 1] - scores[1, 0]])
        # Several rendered answers can tokenize identically; ties are not wins.
        exact = [bool((scores[i, i] - torch.cat([scores[i, :i], scores[i, i+1:]])).min() > 1e-6) for i in range(2)]
        b = buckets.setdefault(r['category'] + '/' + r['axis'], dict(pairs=0, both=0, exact_both=0, wins=0, query_cosine=[]))
        b['pairs'] += 1; b['both'] += int((gap > 1e-6).all()); b['wins'] += int((gap > 1e-6).sum())
        b['exact_both'] += int(all(exact)); b['query_cosine'].append(float(q[0] @ q[1]))
    for b in buckets.values(): b['query_cosine'] = sum(b['query_cosine']) / len(b['query_cosine'])
    return dict(buckets=buckets, texts=len(texts), unknown_tokens=int((tokens['input_ids'] == tk.unk_token_id).sum()),
                max_tokens=int(tokens['attention_mask'].sum(-1).max()), truncations=int((tokens['attention_mask'].sum(-1) == 128).sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'diagnosis.json'
    if target.exists(): raise FileExistsError('Preserve consumed diagnostic')
    torch.set_num_threads(2); device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    rows = probes(); source = torch.load(CURRENT, map_location='cpu', weights_only=False); tk = load_tokenizer(source['tokenizer'])
    previous = json.loads((ROOT / 'training/distill/cache/mini-fa-en-reasoning/texts.json').read_text())['train']
    seen = {r['anchor'] for r in previous}
    assert all(q in seen for r in rows if r['axis'] == 'seen' and r['category'].startswith('numeric') for q in r['queries'])
    result = dict(protocol='One factor changed at a time. Development only; measured before new training.', models={})
    paths = dict(domain=SOURCE, raw=ROOT / 'training/distill/runs/mini-fa-en-reasoning/diagnostic-best.pt', current=CURRENT)
    for name in ['domain', 'raw', 'current', 'current_fp32_embedding', 'current_no_position']:
        path = paths.get(name, CURRENT); checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = load_model(checkpoint, device, train=name == 'current_fp32_embedding').eval()
        if name == 'current_no_position': model.position_embedding.weight.data.zero_()
        result['models'][name] = dict(checkpoint_sha256=digest(path), **score(model, rows, tk, device))
        print(name, json.dumps(result['models'][name]), flush=True)
    (OUT / 'diagnostic-cases.json').write_text(json.dumps(rows, ensure_ascii=False) + '\n')
    target.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': main()
