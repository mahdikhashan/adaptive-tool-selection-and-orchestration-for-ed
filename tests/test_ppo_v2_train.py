"""Pure math/serialization checks. No optimizer or controller training on CPU."""
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from fedorch.ppo_v2_train import (duration_gae, masked_distribution, ppo_clip_loss,
    audit_cost, proximal_term, Rollout, update, metrics, parser, evaluation_missions, mission_items, merge_rollouts)


class LearnerMathTests(unittest.TestCase):
    def test_duration_discount_and_terminal_boundary(self):
        adv, ret = duration_gae([1.,2.,7.],[.5,.6,.7,0.],[2.,3.,1.],
                                [False,True,True], kappa=.2, lam=.95)
        self.assertAlmostEqual(adv[1],1.4)
        self.assertAlmostEqual(adv[0],1+math.exp(-.4)*.6-.5+math.exp(-.4)*.95*1.4)
        self.assertAlmostEqual(ret[2],7.)

    def test_stored_mask_gives_identical_log_probability_and_zero_gradient(self):
        logits=torch.tensor([[1.,100.,2.]], requires_grad=True)
        mask=torch.tensor([[True,False,True]])
        old=masked_distribution(logits.detach(),mask).log_prob(torch.tensor([0]))
        loss,_=ppo_clip_loss(logits,mask,torch.tensor([0]),old,torch.tensor([1.]))
        self.assertAlmostEqual(float(loss),-1.)
        loss.backward()
        self.assertEqual(float(logits.grad[0,1]),0.)
        with self.assertRaises(ValueError): masked_distribution(logits,torch.zeros_like(mask))

    def test_stale_update_rejected_before_any_training(self):
        rollout=Rollout([],[],[],[],[],[],[],[],1)
        with self.assertRaisesRegex(ValueError,'Stale'):
            update(None,None,rollout,2,SimpleNamespace(device='cpu'))

    def test_training_rejects_cpu_before_optimizer(self):
        rollout=Rollout([],[],[],[],[],[],[],[],1)
        with self.assertRaisesRegex(ValueError,'CUDA-only'):
            update(None,None,rollout,1,SimpleNamespace(device='cpu'))

    def test_fedprox_proximal_term(self):
        actor=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): actor.weight.fill_(3.)
        self.assertAlmostEqual(float(proximal_term(actor,{'weight':torch.tensor([[1.]])},.5)),1.)

    def test_flower_delta_and_server_optimizers(self):
        import numpy as np
        from fedorch.ppo_v2_flower import encode_delta, _server_step
        reference=[np.array([1.,2.],dtype=np.float32)]
        delta=encode_delta({'w':torch.tensor([3.,5.])},reference)
        self.assertEqual(delta[0].dtype,np.float16)
        self.assertEqual(delta[0].tolist(),[2.,3.])
        step=lambda opt,lr: SimpleNamespace(server_opt=opt,server_lr=lr,server_momentum=.5,first_moment=None,second_moment=None)
        d=[np.array([2.])]
        self.assertEqual(_server_step(step('none',1.),d)[0].tolist(),[2.])
        m=step('momentum',1.); _server_step(m,d)
        self.assertEqual(_server_step(m,d)[0].tolist(),[3.])
        a=step('adam',.1)
        self.assertAlmostEqual(float(_server_step(a,d)[0][0]),.1*.2/(np.sqrt(.04)+1e-3),places=6)

    def test_supervised_label_guard_known_cost_not_weighted(self):
        self.assertEqual(audit_cost(2,None,False,.05),2)
        self.assertEqual(audit_cost(2,10,True,.05),202)
        with self.assertRaises(ValueError): audit_cost(2,10,False,.05)

    def test_deadline_masks_late_accepted_predictions(self):
        rows=[dict(item_id='a',prediction='rust',abstained=False,within_deadline=False,elapsed_s=7),
              dict(item_id='b',prediction='good',abstained=False,within_deadline=True,elapsed_s=1)]
        report=metrics(rows,{'a':'rust','b':'good'})
        self.assertEqual(report['binary_defect_f1'],1.)
        self.assertEqual(report['on_time']['binary_defect_f1'],0.)
        self.assertEqual(report['deadline_success'],.5)

    def test_grouping_never_fabricates_or_silently_drops_items(self):
        items=[dict(item_id=str(i),tower='t',source_id=str(i)) for i in range(11)]
        groups,excluded=evaluation_missions(items,15)
        self.assertFalse(excluded)
        self.assertEqual(sum(map(len,groups)),11)
        self.assertTrue(all(2<=len(g)<=5 for g in groups))
        self.assertEqual(evaluation_missions(items[:1],15)[1],['0'])
        import random
        self.assertEqual(len(mission_items(items,2,random.Random(0))),1)
        self.assertTrue(2<=len(mission_items(items,15,random.Random(0)))<=5)
        with self.assertRaisesRegex(ValueError,'distinct-source'):
            mission_items(items[:1],15,random.Random(0))

    def test_merge_completed_client_batches_preserves_terminal_boundaries(self):
        a=Rollout([[1]],[[True]],[0],[-.2],[1.],[2.],[True],[.1,0.],3)
        b=Rollout([[2]],[[True]],[0],[-.3],[2.],[1.],[True],[.4,0.],3)
        merged=merge_rollouts([a,b],3)
        self.assertEqual(merged.values,[.1,.4,0.])
        self.assertEqual(merged.dones,[True,True])
        self.assertEqual(merged.states,[[1],[2]])
        with self.assertRaisesRegex(ValueError,'stale'): merge_rollouts([a],4)

    def test_pilot_defaults(self):
        args=parser().parse_args(['--bundle','fixture','--output','fixture'])
        self.assertEqual((args.device,args.rounds,args.episodes_per_client,args.seed),('cuda',3,64,0))
        self.assertEqual(args.eval_split,'val')

if __name__=='__main__': unittest.main()
