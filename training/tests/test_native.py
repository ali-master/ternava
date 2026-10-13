import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/distill'))
from native import persian_rows, int4_ste, retrieval_loss
from prep.native import qa_fixture


class NativeTests(unittest.TestCase):
    def test_mask_preserves_latin_digits_special_and_mixed_tokens(self):
        vocab={t:i for i,t in enumerate(['[PAD]','[UNK]','hello','##ing','۱۲','؟','فارسی','##تر','aب','é'])}
        self.assertEqual(torch.where(persian_rows(vocab))[0].tolist(),[6,7])

    def test_optimizer_and_int4_leave_frozen_rows_exact(self):
        torch.manual_seed(8);layer=torch.nn.Embedding(5,8,padding_idx=0)
        original=layer.weight.detach().clone();mask=torch.tensor([0,0,0,1,1.])[:,None]
        layer.weight.register_hook(lambda g:g*mask);layer.register_forward_hook(int4_ste)
        opt=torch.optim.AdamW(layer.parameters(),lr=.02,weight_decay=0)
        for _ in range(3):
            opt.zero_grad();layer(torch.arange(5)).sum().backward();opt.step()
        self.assertTrue(torch.equal(layer.weight[:3],original[:3]))
        self.assertFalse(torch.equal(layer.weight[3:],original[3:]))
        layer.eval();self.assertTrue(torch.equal(layer(torch.arange(5)),layer.weight))

    def test_duplicate_answers_are_not_negatives(self):
        v=F.normalize(torch.randn(3,2,8),dim=-1).requires_grad_()
        self.assertAlmostEqual(float(retrieval_loss(v,torch.ones(3,3,dtype=torch.bool)).detach()),0)
        loss=retrieval_loss(v,torch.eye(3,dtype=torch.bool));loss.backward()
        self.assertTrue(torch.isfinite(v.grad).all())
        self.assertGreater(float(loss.detach()),0)

    def test_fake_quantization_matches_rowwise_export_and_keeps_gradient(self):
        layer=torch.nn.Embedding(3,4)
        x=torch.tensor([[.001,-.0005,.0002,0.],[1e-10,-1e-10,0.,0.]],requires_grad=True)
        out=int4_ste(layer,None,x)
        scale=(x.detach().abs().amax(-1,keepdim=True)/7).clamp_min(1e-8)
        self.assertTrue(torch.allclose(out,(x/scale).round().clamp(-7,7)*scale,atol=1e-12))
        out.sum().backward();self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))

    def test_qa_windows_are_answer_independent_and_reject_bad_offsets(self):
        class Tokenizer:
            def __call__(self,text,**kwargs):return {'offset_mapping':[(i,i+1) for i in range(len(text))]}
        text='الف '*60
        q=dict(question='چه؟',answers=[dict(answer_start=2,text=text[2:4])])
        a=[dict(paragraphs=[dict(context=text,qas=[q])])]
        f=qa_fixture(a,Tokenizer())
        q['answers']=[dict(answer_start=130,text=text[130:132])]
        g=qa_fixture(a,Tokenizer())
        self.assertEqual(f['corpus'],g['corpus'])
        self.assertNotEqual(f['queries'][0]['relevant'],g['queries'][0]['relevant'])


if __name__=='__main__':unittest.main()
