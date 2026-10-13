import json
import random
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/distill'))
from compositional import objective
from diagnose_reasoning import numeric_probe
from prep.compositional import CHALLENGE, KEYS, generate


class CompositionalTests(unittest.TestCase):
    def test_factorial_probe_changes_one_axis(self):
        state=random.Random(33).getstate()
        a=numeric_probe(state,'جعبه',1,0,'seen')
        wording=numeric_probe(state,'جعبه',1,0,'wording')
        masked=numeric_probe(state,'جعبه',1,0,'no_numbers')
        changed=numeric_probe(state,'جعبه',1,0,'range')
        self.assertEqual(a['values'],wording['values']);self.assertEqual(a['docs'],wording['docs'])
        self.assertEqual(a['docs'],masked['docs']);self.assertNotEqual(a['queries'],masked['queries'])
        self.assertEqual([x+400 for x in a['values']],changed['values'])
        self.assertNotEqual(a['queries'],wording['queries'])

    def test_truth_table_and_unknown_evidence(self):
        rows=[r for r in generate('train',120) if r['category']=='negation']
        for r in rows:
            expected=r['base']
            for op in r['operators']:expected^=op
            if r['state']==2:
                self.assertIn('نامعلوم',r['positive'])
                self.assertTrue(any(s in r['query'] for s in ['هیچ نتیجه','نه این گفته','هر دو امکان']))
            else:self.assertEqual(expected,r['state'])
            self.assertEqual(r['signature'][0][-1],r['state'])
        self.assertEqual({r['state'] for r in rows},{0,1,2})

    def test_numeric_negatives_include_neighbors_and_same_semantic_domain(self):
        rows=[r for r in generate('train',100) if r['category']=='numbers']
        for r in rows:
            values=[s[-1] for s in r['signature']]
            x,y,z,sign=r['operands'];family=r['family']
            expected=[lambda:x+sign*y,lambda:x-y+sign*z,lambda:x*(100+sign*y)//100,
                      lambda:x//y*(z if sign==1 else y-z),lambda:sign*y,lambda:y,
                      lambda:int(x>=y) if sign==1 else int(x>y),lambda:x*y+sign*z][family]()
            self.assertEqual(r['value'],expected)
            self.assertEqual(values[0],r['value']);self.assertEqual(len(set(values)),4)
            self.assertTrue(any(abs(v-values[0])==1 for v in values[1:]))
            self.assertEqual(len({s[1] for s in r['signature']}),1)
            self.assertNotEqual(r['positive'],r['negative'])

    def test_distinct_validation_views_and_balanced_boundary_labels(self):
        for split in ['train','val']:
            rows=generate(split,2400 if split=='train' else 160)
            self.assertTrue(all(r['query']!=r['paraphrase'] for r in rows))
            bounds=[r['value'] for r in rows if r['category']=='numbers' and r['family']==6]
            self.assertGreaterEqual(sum(bounds)/len(bounds),.3)
            self.assertLessEqual(sum(bounds)/len(bounds),.7)

    def test_full_corpus_impostor_is_penalized_and_gradients_finite(self):
        good=F.normalize(torch.randn(2,6,12,generator=torch.Generator().manual_seed(9)),dim=-1)
        good[:,1]=good[:,0];good[:,2]=good[:,0]
        ids=torch.tensor([[0,1,2,3],[4,5,6,7]])
        bad=good.clone();bad[1,5]=bad[0,0]
        self.assertAlmostEqual(objective(good,ids,True).item(),objective(bad,ids,True).item())
        self.assertGreater(objective(bad,ids).item(),objective(good,ids).item())
        bad.requires_grad_();objective(bad,ids).backward()
        self.assertTrue(torch.isfinite(bad.grad).all());self.assertGreater(bad.grad.abs().sum().item(),0)

    def test_duplicate_positives_are_not_false_negatives(self):
        v=F.normalize(torch.randn(1,6,12,generator=torch.Generator().manual_seed(10)),dim=-1)
        v[:,1]=v[:,0];v[:,2]=v[:,0]
        once=objective(v,torch.tensor([[0,1,2,3]]))
        twice=objective(v.repeat(2,1,1),torch.tensor([[0,1,2,3],[0,1,2,3]]))
        self.assertAlmostEqual(once.item(),twice.item(),places=5)

    def test_fresh_fixture_balance_and_ids(self):
        suites=json.loads(CHALLENGE.read_text())['suites']
        self.assertEqual(sum(len(s['queries']) for s in suites.values()),64)
        for name,s in suites.items():
            ids={d['id'] for d in s['corpus']}
            self.assertEqual(len(ids),len(s['corpus']))
            for q in s['queries']:self.assertTrue(set(q['relevant'])<=ids)
            if name.startswith(('fa_','en_')):
                for a,b in zip(s['queries'][::2],s['queries'][1::2]):
                    self.assertEqual(a['pair'],b['pair']);self.assertEqual(set(a['relevant']),{b['contrast']})


if __name__=='__main__':unittest.main()
