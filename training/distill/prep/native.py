"""Prepare a round-three continuation from pinned Persian QA, QQP and NLI labels."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

import torch

from prep.round2 import ROOT, encode_rows, normalize
from tokenization import load_tokenizer

CACHE = ROOT / 'training/distill/cache/mini-fa-en-round3-native'
SOURCES = ROOT / 'training/distill/cache/persian-native-sources'
SOURCE = ROOT / 'training/distill/runs/mini-fa-en-round3/best.pt'


def fetch_sources():
    sources = [
        ('sajjjadayobi/PersianQA','ba4806eb7894b1e78e214e16e7e717840c129450',
         {'LICENSE':'persianqa/LICENSE', **{f'dataset/pqa_{s}.json':f'persianqa/pqa_{s}.json' for s in ['train','test']}}),
        ('persiannlp/parsinlu','a3b08d0b93ca84e0017e2f490a2aa3ab36c79885',
         {p:p for p in ['LICENSE']+[f'data/qqp/{s}.jsonl' for s in ['train','dev','test']]+
          [f'data/entailment/merged_with_farstail/{s}.tsv' for s in ['train','dev','test_natural']]})]
    for repo, revision, files in sources:
        for remote, local in files.items():
            path=SOURCES/local
            if path.exists():continue
            with urlopen(f'https://raw.githubusercontent.com/{repo}/{revision}/{remote}',timeout=60) as response:
                content=response.read()
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def windows(text, tokenizer):
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        padding=False, return_offsets_mapping=True)['offset_mapping']
    return [(offsets[i][0], offsets[min(i + 95, len(offsets)-1)][1])
            for i in range(0, len(offsets), 64)]


def qa_fixture(articles, tokenizer):
    corpus, queries = [], []
    skipped = defaultdict(int)
    for article in articles:
        for para in article['paragraphs']:
            text = para['context']
            chunks = windows(text, tokenizer)
            ids = []
            for start, end in chunks:
                doc_id = str(len(corpus)); ids.append(doc_id)
                corpus.append(dict(id=doc_id, text=text[start:end], lang='fa'))
            for qa in para['qas']:
                if qa.get('is_impossible') or not qa['answers']:
                    skipped['unanswerable'] += 1; continue
                spans = []
                for answer in qa['answers']:
                    start = answer['answer_start']; end = start + len(answer['text'])
                    if text[start:end] == answer['text']: spans.append((start, end))
                relevant = {ids[i]: 1 for i, (s, e) in enumerate(chunks)
                            if any(s <= a and b <= e for a, b in spans)}
                if not relevant:
                    skipped['invalid_or_uncovered_answer_span'] += 1; continue
                queries.append(dict(id=str(len(queries)), text=qa['question'],
                                    lang='fa', relevant=relevant, category='native-qa'))
    assert corpus and queries
    return dict(corpus=corpus, queries=queries, skipped=dict(skipped))


def prepare():
    if CACHE.exists(): raise FileExistsError('Preserve prepared native experiment')
    fetch_sources()
    source = torch.load(SOURCE, map_location='cpu', weights_only=False)
    tk = load_tokenizer(source['tokenizer'])
    articles = json.loads((SOURCES/'persianqa/pqa_train.json').read_text())['data']
    test = json.loads((SOURCES/'persianqa/pqa_test.json').read_text())['data']
    test_titles = {normalize(a['title']) for a in test}
    splits = {'train': [], 'val': [], 'test': test}
    for article in articles:
        key = normalize(article['title'])
        if key in test_titles: continue
        split = 'val' if int(hashlib.sha256(key.encode()).hexdigest(),16) % 10 == 0 else 'train'
        splits[split].append(article)
    # Reserve all evaluation endpoints, including inherited regression fixtures.
    fixtures = {s: qa_fixture(a, tk) for s,a in splits.items()}
    reserved = []
    for s in ['val','test']:
        reserved += [r['text'] for r in fixtures[s]['corpus']+fixtures[s]['queries']]
    for path in (ROOT/'eval/quality/corpus').glob('bilingual-*.json'):
        f = json.loads(path.read_text()); reserved += [r['text'] for r in f['corpus']+f['queries']]
    external = json.loads((ROOT/'dist/pilot-fa-en/external-sts.json').read_text())
    reserved += [r[k] for b in external['benchmarks'].values() for r in b['samples']
                 for k in ['sentence1','sentence2']]
    forbidden = {normalize(t) for t in reserved}
    data, texts = {}, {}
    for split, fixture in fixtures.items():
        lookup = {d['id']: d['text'] for d in fixture['corpus']}
        rows = [dict(anchor=q['text'], positive=lookup[next(iter(q['relevant']))],
                     group=tuple(sorted(q['relevant']))) for q in fixture['queries']]
        if split == 'train':
            rows = [r for r in rows if not any(normalize(r[k]) in forbidden for k in ['anchor','positive'])]
        texts['qa_'+split] = rows
        if split != 'test':
            encoded = encode_rows(rows, ['anchor','positive'], tk)
            # Overlapping windows from the same paragraph are not false negatives.
            groups = {}
            for i,r in enumerate(rows):
                for doc in r['group']: groups.setdefault(doc, set()).add(i)
            mask = torch.eye(len(rows), dtype=torch.bool)
            for indices in groups.values():
                ix = torch.tensor(sorted(indices)); mask[ix[:,None],ix[None,:]] = True
            encoded['same_positive'] = mask
            data['qa_'+split] = encoded
    for split, filename in [('train','train'),('val','dev'),('test','test')]:
        rows = [json.loads(line) for line in (SOURCES/f'data/qqp/{filename}.jsonl').read_text().splitlines()]
        if split == 'train':
            for held in ['dev','test']:
                forbidden.update(normalize(r[k]) for line in (SOURCES/f'data/qqp/{held}.jsonl').read_text().splitlines()
                                 for r in [json.loads(line)] for k in ['q1','q2'])
            rows = [r for r in rows if not any(normalize(r[k]) in forbidden for k in ['q1','q2'])]
        texts['qqp_'+split] = rows
        if split != 'test':
            data['qqp_'+split] = encode_rows(rows,['q1','q2'],tk)
            data['qqp_'+split]['label'] = torch.tensor([int(r['label']) for r in rows])
    nli = {}
    for split, filename in [('train','train'),('val','dev'),('test','test_natural')]:
        groups = defaultdict(lambda: defaultdict(list))
        for line in (SOURCES/f'data/entailment/merged_with_farstail/{filename}.tsv').read_text().splitlines():
            pair, label = line.rsplit('\t',1)
            if label not in ['e','c']: continue
            a,h = pair.split('<sep>',1); groups[a][label].append(h)
        nli[split] = [dict(anchor=a,positive=g['e'][0],negative=g['c'][0])
                      for a,g in groups.items() if g['e'] and g['c']]
    forbidden.update(normalize(r[k]) for s in ['val','test'] for r in nli[s] for k in r)
    for split, rows in nli.items():
        rows = [r for r in rows if normalize(r['positive']) != normalize(r['negative']) and
                (split!='train' or not any(normalize(t) in forbidden for t in r.values()))]
        texts['nli_'+split] = rows
        if split!='test': data['nli_'+split] = encode_rows(rows,['anchor','positive','negative'],tk)
    # Stronger exact-endpoint audit across all three tasks, not just within each source.
    held = {normalize(t) for key,rows in texts.items() if not key.endswith('train')
            for r in rows for k,t in r.items() if k in ['anchor','positive','negative','q1','q2']}
    for key in ['qa_train','qqp_train','nli_train']:
        keys = ['anchor','positive'] if key.startswith('qa') else ['q1','q2'] if key.startswith('qqp') else ['anchor','positive','negative']
        keep = torch.tensor([i for i,r in enumerate(texts[key]) if not any(normalize(r[k]) in held for k in keys)])
        texts[key] = [texts[key][i] for i in keep.tolist()]
        data[key] = {k:(v[keep][:,keep] if k=='same_positive' else v[keep]) for k,v in data[key].items()}
        assert len(keep)>0
    CACHE.mkdir(parents=True)
    for s,f in fixtures.items(): (CACHE/f'qa-{s}.json').write_text(json.dumps(f,ensure_ascii=False)+'\n')
    (CACHE/'texts.json').write_text(json.dumps(texts,ensure_ascii=False)+'\n')
    torch.save(data,CACHE/'data.pt')
    manifest = dict(tokenizer=source['tokenizer'], source_sha256=digest(SOURCE),
        counts={k:len(v) for k,v in texts.items()}, articles={s:len(a) for s,a in splits.items()},
        source_files={str(p.relative_to(SOURCES)):digest(p) for p in SOURCES.rglob('*') if p.is_file()},
        revisions={'PersianQA':'ba4806eb7894b1e78e214e16e7e717840c129450','ParsiNLU':'a3b08d0b93ca84e0017e2f490a2aa3ab36c79885'},
        licenses={'PersianQA':'GPL-3.0','ParsiNLU':'CC-BY-NC-SA-4.0'},
        split='Official test untouched; train/dev QA separated by normalized article title SHA modulo 10; 96-token windows every 64 tokens independent of answers; exact held-out endpoints removed across tasks',
        caveat='Public held-out data may have appeared in ancestor teacher pretraining; this experiment only audits its own supervised inputs',
        files={p.name:digest(p) for p in CACHE.iterdir() if p.is_file()})
    (CACHE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['counts','articles']},indent=2))


if __name__ == '__main__': prepare()
