"""CPU tests of the new gradient path, fixed teacher, mixed-precision master and restart."""
import copy, json, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import config
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from encbank import Encbank
import torch.nn.functional as F
import train_support as T
from train_full import FullReader, full_loss, MasterAdam


def tiny():
    torch.manual_seed(23)
    cfg=Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=256,attention_dropout=0.)
    cfg._attn_implementation='sdpa'
    return Qwen3ForCausalLM(cfg).float().eval()


ARGS=SimpleNamespace(chunk=8,n_ctx=2,topk=16,lam=.6,logit_chunk=3,loss='published')
WINDOW=torch.arange(24)+3


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_matches_original_packing_and_loss(self):
        model=tiny();new=FullReader(model,2);old=T.PublishedReader(model,2)
        for teacher in (False,True):
            with torch.no_grad():
                got=model.lm_head(new.hidden(WINDOW,1,8,2,teacher))
                expected=model.lm_head(old.hidden(WINDOW,1,8,2,teacher))
                torch.testing.assert_close(got,expected,rtol=0,atol=0)
                cm=Encbank(model,0 if teacher else 2)
                seg=WINDOW.split(8)
                ref=cm.read_core(cm.write_chunk([1]),[cm.write_chunk(c) for c in seg[:-1]],cm.write_chunk(seg[-1]),logits_tail=8)
                torch.testing.assert_close(got,ref,atol=2e-6,rtol=2e-5)
        val,idx=torch.randn(3,31).topk(9,dim=-1)
        a=torch.randn(3,31,requires_grad=True);b=a.detach().clone().requires_grad_()
        x=T.support_loss(a,idx,val,.6,'published')
        logp=F.log_softmax(val,dim=-1);logq=F.log_softmax(b.gather(-1,idx),dim=-1)
        y=.6*F.kl_div(logq,logp,reduction='batchmean',log_target=True)+.4*F.kl_div(logp,logq,reduction='batchmean',log_target=True)
        x.backward();y.backward()
        torch.testing.assert_close(x,y);torch.testing.assert_close(a.grad,b.grad)

    def test_all_student_gradients_checkpoint_parity_and_teacher_fixed(self):
        teacher=tiny().requires_grad_(False)
        before=copy.deepcopy(teacher.state_dict())
        a=tiny();b=copy.deepcopy(a)
        la=full_loss(FullReader(a,2,True),T.PublishedReader(teacher,2),WINDOW,1,ARGS)
        lb=full_loss(FullReader(b,2,False),T.PublishedReader(teacher,2),WINDOW,1,ARGS)
        la.backward();lb.backward();torch.testing.assert_close(la,lb)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.abs().sum()),0,name)
            torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=1e-4)
        opt=MasterAdam(a,1e-3); opt.step(1e-3)
        self.assertFalse(torch.equal(a.model.layers[0].self_attn.q_proj.weight,teacher.model.layers[0].self_attn.q_proj.weight))
        for name,p in teacher.named_parameters():
            self.assertIsNone(p.grad);torch.testing.assert_close(p,before[name],atol=0,rtol=0)

    def test_master_optimizer_restart_exact(self):
        teacher=tiny().requires_grad_(False)
        a=tiny();opt=MasterAdam(a,1e-3)
        def advance(model,optim):
            optim.zero_grad();full_loss(FullReader(model,2),T.PublishedReader(teacher,2),WINDOW,1,ARGS).backward();optim.step(1e-3)
        advance(a,opt)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'restart.pt'
            torch.save(dict(model=a.state_dict(),master=opt.state_dict()),path)
            saved=torch.load(path,weights_only=False)
            b=tiny();b.load_state_dict(saved['model']);other=MasterAdam(b,1e-3);other.load_state_dict(saved['master'])
        advance(a,opt);advance(b,other)
        for p,q in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        for p,q in zip(opt.master,other.master):
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        # BF16 student updates retain sub-quantization changes in FP32 masters.
        linear=torch.nn.Linear(4,4,bias=False).to(torch.bfloat16)
        master=MasterAdam(linear,1e-5);start=master.master[0].detach().clone()
        linear.weight.grad=torch.ones_like(linear.weight);master.step(1e-5)
        self.assertEqual(master.master[0].dtype,torch.float32)
        self.assertFalse(torch.equal(master.master[0],start))
        torch.testing.assert_close(linear.weight,master.master[0].to(torch.bfloat16),rtol=0,atol=0)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    T.atomic_json(config.ROOT/'cpu_checks.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),torch=torch.__version__))
    raise SystemExit(0 if result.wasSuccessful() else 1)
