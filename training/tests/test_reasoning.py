import json
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/distill'))
from reasoning import paired_loss, geometry_loss, eligible
from reasoning_calibrate import language_mask
from prep.reasoning import synthetic, CHALLENGE


class ReasoningTests(unittest.TestCase):
    def test_worst_direction_penalizes_constant_answer_and_has_gradients(self):
        good=torch.tensor([[[1.,0.],[1.,0.],[0.,1.]],[[0.,1.],[0.,1.],[1.,0.]]],requires_grad=True)
        fixed=good.detach().clone();fixed[1,0]=fixed[0,0];fixed.requires_grad_()
        self.assertLess(paired_loss(good).item(),paired_loss(fixed).item())
        paired_loss(fixed).backward();self.assertTrue(torch.isfinite(fixed.grad).all());self.assertGreater(fixed.grad.abs().sum().item(),0)

    def test_geometry_loss_penalizes_changed_neighborhood(self):
        base=torch.eye(3);moved=base.clone();moved[0]=base[1]
        self.assertLess(abs(geometry_loss(base,base).item()),1e-5)
        self.assertGreater(geometry_loss(moved,base).item(),1)
        with self.assertRaises(AssertionError):geometry_loss(base[:,0],base[:,0])

    def test_generator_keeps_both_directions_and_distinct_answers(self):
        rows=synthetic('train',30)
        for a,b in zip(rows[::2],rows[1::2]):
            self.assertEqual(a['positive'],b['negative']);self.assertEqual(a['negative'],b['positive'])
            self.assertNotEqual(a['anchor'],b['anchor']);self.assertNotEqual(a['positive'],a['negative'])

    def test_fresh_fixture_has_64_valid_queries_and_balanced_pairs(self):
        f=json.loads(CHALLENGE.read_text())['suites'];self.assertEqual(sum(len(s['queries']) for s in f.values()),64)
        for name,s in f.items():
            ids={d['id'] for d in s['corpus']}
            for q in s['queries']:self.assertTrue(set(q['relevant'])<=ids)
            if 'negation' in name or 'numbers' in name:
                for a,b in zip(s['queries'][::2],s['queries'][1::2]):
                    self.assertEqual(a['pair'],b['pair']);self.assertEqual(set(a['relevant']),{b['contrast']})

    def test_english_safe_mask_excludes_latin_and_shared_digits(self):
        vocab={t:i for i,t in enumerate(['[PAD]','hello','##123','۱۲۳','##٣','منفی','##one','1۲'])}
        self.assertEqual(language_mask(vocab).tolist(),[False,False,False,True,True,True,False,False])

    def test_every_english_ranking_set_must_recover(self):
        b=dict(sts_en=.8,sts_fa=.7,negation=.6,numbers=.5,fa_negation_paired=.4,fa_numbers_paired=.2)
        b.update({f'r{i}_{l}_ndcg':.7 for i in [2,3,4] for l in ['fa','en']})
        b.update({'domain_'+k+'_top1':.6 for k in ['docs','emoji','unseen_corpus']})
        self.assertTrue(eligible(b,b,b))
        self.assertFalse(eligible({**b,'r2_en_ndcg':.69},b,b))
        self.assertFalse(eligible({**b,'domain_emoji_top1':.59},b,b))


if __name__=='__main__':unittest.main()
