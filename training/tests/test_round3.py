import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training/distill'))
from round3 import ranking_loss, validation_fixture
from prep.round3 import numeral, synthetic, normalize


class RoundThreeTests(unittest.TestCase):
    def test_listwise_loss_penalizes_unrelated_document_ahead_of_gold(self):
        vectors=F.normalize(torch.tensor([[[1.,0.,0.],[.8,.6,0.],[0.,1.,0.]],[[0.,0.,1.],[1.,0.,.1],[0.,0.,1.]]]),dim=-1).requires_grad_()
        topics=torch.tensor([0,1]);ids=torch.tensor([[0,1],[2,3]]);groups=torch.tensor([0,1])
        loss=ranking_loss(vectors,topics,ids,groups)
        loss.backward()
        self.assertTrue(torch.isfinite(vectors.grad).all())
        # Query zero beats its own contradiction, but another topic is still ahead.
        better=vectors.detach().clone();better[0,1]=torch.tensor([1.,0.,0.])
        self.assertLess(ranking_loss(better,topics,ids,groups),loss)

    def test_duplicate_gold_and_same_topic_are_not_false_negatives(self):
        vectors=torch.tensor([[[1.,0.],[1.,0.],[0.,1.]],[[1.,0.],[1.,0.],[0.,1.]]])
        single=ranking_loss(vectors[:1],torch.tensor([0]),torch.tensor([[1,2]]),torch.tensor([0]))
        duplicate=ranking_loss(vectors,torch.tensor([0,0]),torch.tensor([[1,2],[1,2]]),torch.tensor([0,0]))
        torch.testing.assert_close(single,duplicate)

    def test_numeric_forms_and_validation_document_deduplication(self):
        self.assertEqual(numeral(240,'fa',0),'دویست و چهل')
        self.assertEqual(numeral(-16,'en',0),'minus sixteen')
        self.assertEqual(numeral(315,'fa',2),'۳۱۵')
        rows=synthetic('val',600)
        self.assertTrue(all(normalize(r['positive'])!=normalize(r['negative']) for r in rows))
        for lang in ['fa','en']:
            self.assertEqual({r['template'] for r in rows if r['lang']==lang and r['category']=='numbers'},set(range(8)))
        fixture=validation_fixture(rows)
        keys=[(d['lang'],normalize(d['text'])) for d in fixture['corpus']]
        self.assertEqual(len(keys),len(set(keys)))
        for q in fixture['queries']:
            self.assertNotEqual(q['positive'],q['negative'])
            self.assertEqual(fixture['corpus'][q['positive']]['lang'],q['lang'])

    def test_not_all_rejected_does_not_assert_approval(self):
        rows=[r for r in synthetic('train',600) if r['category']=='negation' and r['template']==1]
        self.assertTrue(rows)
        for r in rows:
            for key in ['positive','negative']:
                self.assertNotIn('approved',r[key])
                self.assertNotIn('تایید',r[key])

    def test_permission_examples_state_sufficiency_not_only_necessity(self):
        rows=[r for r in synthetic('train',600) if r['category']=='negation' and r['template']==7]
        self.assertTrue(rows)
        for r in rows:
            self.assertIn('only condition' if r['lang']=='en' else 'تنها شرط',r['anchor'])


if __name__=='__main__':unittest.main()
