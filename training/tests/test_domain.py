import json
import importlib.util
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training/distill'))
from domain import listwise_loss, filter_triplets, eligible, replay_groups
from prep.domain import domain_rows, CHALLENGE
from prep.round2 import normalize


class DomainTests(unittest.TestCase):
    def test_constant_choice_does_not_pass_counterfactual_pair(self):
        root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'eval/quality'))
        spec=importlib.util.spec_from_file_location('domain_eval',root/'eval/quality/domain.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        fixture=dict(corpus=[dict(id='a',text='A'),dict(id='b',text='B')],queries=[
            dict(id='q1',text='Q1',category='numbers',pair=0,relevant={'a':1},contrast='b'),
            dict(id='q2',text='Q2',category='numbers',pair=0,relevant={'b':1},contrast='a')])
        vectors=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[1.,0.]]).numpy()
        result=module.score_fixture(vectors,{t:i for i,t in enumerate(['A','B','Q1','Q2'])},fixture)
        self.assertEqual(result['categories']['numbers']['contrast_accuracy'],.5)
        self.assertEqual(result['pairs']['numbers']['both_contrasts_correct'],0)

    def test_replay_uses_source_language_not_row_position(self):
        pairs=[dict(en='hello',fa='سلام'),dict(en='thanks',fa='ممنون')]
        groups=replay_groups(['سلام','thanks','ممنون','hello','English NLI'],pairs,opus_count=4)
        self.assertEqual(groups[0].tolist(),[1,3,4]);self.assertEqual(groups[1].tolist(),[0,2])

    def test_multiple_correct_emoji_are_not_false_negatives(self):
        q=torch.eye(2,requires_grad=True);d=torch.eye(2)
        self.assertEqual(float(listwise_loss(q,d,torch.ones(2,2,dtype=torch.bool)).detach()),0)
        loss=listwise_loss(q,d,torch.eye(2,dtype=torch.bool));loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertLess(float(loss.detach()),.001)

    def test_filter_removes_both_counterfactuals_if_one_collides(self):
        data={'input_ids':torch.tensor([[[1],[2],[2]],[[3],[4],[5]],[[6],[7],[8]],[[9],[10],[11]]]),'group':torch.arange(4)}
        texts=[dict(anchor=str(i),positive='x',negative='y') for i in range(4)]
        self.assertEqual(filter_triplets(data,texts,set(),paired=True)['group'].tolist(),[2,3])
        self.assertEqual(filter_triplets(data,texts,{'2'})['group'].tolist(),[1,3])

    def test_retention_gates_cannot_be_hidden_by_domain_gain(self):
        keys=['sts_en','sts_fa','negation','numbers']+[f'r{i}_{l}_ndcg' for i in [2,3,4] for l in ['fa','en']]+['previous_'+k+'_top1' for k in ['docs','emoji','odyssey']]
        base=dict.fromkeys(keys,.7)
        self.assertTrue(eligible(base,base))
        for key in keys:
            changed={**base,key:.67,'selection_score':1.}
            self.assertFalse(eligible(changed,base),key)

    def test_fresh_queries_and_logic_pairs_are_separate_and_valid(self):
        corpora,rows=domain_rows();f=json.loads(CHALLENGE.read_text())['suites']
        seen={normalize(q['text']) for s in rows.values() for q in s}
        self.assertEqual(len(rows['train']),300);self.assertEqual(len(rows['val']),100)
        for task,fixture in f.items():
            ids={d['id'] for d in fixture['corpus']}
            for q in fixture['queries']:
                self.assertNotIn(normalize(q['text']),seen)
                self.assertTrue(set(q['relevant'])<=ids)
        logic=f['logic']['queries']
        self.assertEqual(len(logic),24)
        for a,b in zip(logic[::2],logic[1::2]):
            self.assertEqual(a['pair'],b['pair'])
            self.assertNotEqual(a['relevant'],b['relevant'])
            self.assertEqual(set(a['relevant']),{b['contrast']})
        self.assertEqual(len(corpora['docs']),30);self.assertEqual(len(corpora['emoji']),70)


if __name__=='__main__':unittest.main()
