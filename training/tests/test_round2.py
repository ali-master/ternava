import sys,tempfile,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'training/distill'),str(ROOT/'training/distill/prep'),str(ROOT/'training/pack')]
from model import StudentEncoder
from pack import pack
from unpack import UnpackedModel
from verify import _load_source_model
from format import Header
from round2 import triplet_loss,cosent_loss

class RoundTwoTests(unittest.TestCase):
    def test_positions_break_role_permutation_and_survive_pack(self):
        torch.manual_seed(51);torch.set_num_threads(2)
        cfg=dict(vocab_size=30522,d_model=16,n_layers=1,n_heads=2,ffn_dim=32,output_dim=384,dropout=0.0)
        ids=torch.tensor([[1,3,4,5,2],[1,5,4,3,2]]);mask=torch.ones_like(ids)
        legacy=StudentEncoder(**cfg).eval()
        torch.testing.assert_close(*legacy(ids,mask),atol=1e-6,rtol=1e-5)
        cfg['position_embeddings']=True;model=StudentEncoder(**cfg).eval()
        self.assertGreater(float((model(ids,mask)[0]-model(ids,mask)[1]).abs().max()),1e-4)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt=Path(tmp)/'model.pt';out=Path(tmp)/'model.bin'
            torch.save({'model_state':model.state_dict(),'config':dict(cfg,enable_qat=True),'epoch':1},ckpt)
            pack(ckpt,out,'int4');ref=UnpackedModel.from_bin(out);source=_load_source_model(ckpt,'int4')
            self.assertEqual(ref.header.format_version,2)
            torch.testing.assert_close(ref.forward(ids,mask),source(ids,mask),atol=5e-4,rtol=0)
            self.assertEqual(ref.position_embedding.shape,(128,16))
            data=bytearray(out.read_bytes());data[4:6]=(3).to_bytes(2,'little')
            with self.assertRaises(ValueError):Header.unpack(data)
    def test_supervised_losses_follow_labels_and_backpropagate(self):
        correct=torch.tensor([[[1.,0.],[1.,0.],[0.,1.]]],requires_grad=True)
        wrong=correct[:,[0,2,1]]
        self.assertLess(triplet_loss(correct),triplet_loss(wrong))
        triplet_loss(wrong).backward();self.assertTrue(torch.isfinite(correct.grad).all())
        pairs=torch.tensor([[[1.,0.],[1.,0.]],[[1.,0.],[0.,1.]]])
        self.assertLess(cosent_loss(pairs,torch.tensor([1.,0.])),cosent_loss(pairs,torch.tensor([0.,1.])))
        self.assertEqual(float(cosent_loss(pairs,torch.tensor([1.,1.]))),0)
if __name__=='__main__':unittest.main()
