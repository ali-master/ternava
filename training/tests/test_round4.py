import random
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/distill'))
from round4 import contrast_loss, paired_indices, eligible, TARGETS
from prep.round4 import synthetic, numeric, render_number
from prep.round2 import normalize


class RoundFourTests(unittest.TestCase):
    def test_contrast_rejects_tied_answers_even_when_topic_is_correct(self):
        v=F.normalize(torch.tensor([[[1.,0.],[1.,0.],[1.,0.]]]),dim=-1).requires_grad_()
        loss=contrast_loss(v);loss.backward()
        self.assertGreater(float(loss),.06)
        self.assertTrue(torch.isfinite(v.grad).all())
        better=v.detach().clone();better[0,2]=torch.tensor([0.,1.])
        self.assertLess(contrast_loss(better),loss)

    def test_pairs_reverse_answers_and_hold_out_wording(self):
        for split in ['train','val']:
            rows=synthetic(split,80)
            for i in range(0,len(rows),2):
                a,b=rows[i:i+2]
                self.assertEqual(a['positive'],b['negative']);self.assertEqual(a['negative'],b['positive'])
                self.assertNotEqual(normalize(a['positive']),normalize(a['negative']))
                self.assertNotEqual(a['anchor'],b['anchor'])
                self.assertEqual(a['wording']==3,split=='val')
        ids=paired_indices([torch.tensor([0,2]),torch.tensor([4,6])])
        self.assertTrue(torch.equal(ids[1::2],ids[::2]+1))

    def test_four_digit_numeric_answer(self):
        self.assertEqual(render_number(1200,'fa',0),'یک هزار و دویست')
        self.assertEqual(render_number(1080,'en',0),'one thousand and eighty')

    def test_equal_fraction_answers_are_dropped(self):
        # Half sound and half defective must never become a contradictory pair.
        class Fixed(random.Random):
            def choice(self,seq):return 4
            def randrange(self,*args):return 2 if args==(1,4) else 10
        self.assertIsNone(numeric(Fixed(),'frame','en',3,0))

    def test_ranking_cannot_hide_numeric_or_english_regression(self):
        b={**dict.fromkeys(TARGETS,.6),'sts_en':.8,'sts_fa':.8,'r3_fa_ndcg':.8,'r3_en_ndcg':.8}
        self.assertTrue(eligible(b,b))
        for key in ['fa_numbers','en_negation','sts_en','r3_en_ndcg']:
            m={**b,key:b[key]-.1,'fa_ndcg':1,'en_ndcg':1}
            self.assertFalse(eligible(m,b))

if __name__=='__main__':unittest.main()
