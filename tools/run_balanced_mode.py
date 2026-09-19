"""固定等权 Balanced PPO 的 prepare/train/report 阶段闸门。"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from typing import Any, Dict, Sequence
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from resource_management.contract_v1 import contract_digest, verify_frozen
from resource_management.unified_evaluation import FrozenEvaluation, evaluate_episode, validate_records
from rl_resource.ablation import FrozenBaseline, assert_frozen
from rl_resource.train import TrainConfig, train

CFG=os.path.join(ROOT,"config","balanced_protocol_v1.json"); SHA=CFG.replace(".json",".sha256")
OUT=os.path.join(ROOT,"output","rl_resource","balanced")

def digest(path: str)->str:
 h=hashlib.sha256(); h.update(open(path,"rb").read()); return h.hexdigest()
def load(path: str)->Dict[str,Any]: return json.load(open(path,encoding="utf-8"))
def dump(path: str,data: Any)->None:
 os.makedirs(os.path.dirname(path),exist_ok=True); json.dump(data,open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
def contract()->str: return os.path.join(OUT,"balanced_training_freeze.json")
def checked()->Dict[str,Any]:
 cfg=load(CFG); expected=open(SHA).read().strip();
 if digest(CFG)!=expected: raise RuntimeError("Balanced 协议 SHA 不一致")
 if not cfg["test_v3_sealed"] or not verify_frozen(strict=False)["ok"]: raise RuntimeError("test-v3 或资源契约未封存")
 return cfg

def prepare()->None:
 cfg=checked(); payload={"protocol_sha256":digest(CFG),"contract_digest":contract_digest(),"weights":cfg["balanced_weights"],"normalization":cfg["normalization"],"train_validation":cfg["train_validation"],"training_seeds":cfg["training_seeds"],"test_v3":"sealed; no training command reads its contents"}
 dump(contract(),payload); print(json.dumps({"prepared":contract(),"sha256":digest(contract())},ensure_ascii=False))

def train_one(seed:int)->None:
 c=load(contract())
 if seed not in c["training_seeds"]: raise ValueError("未冻结的训练 seed")
 f=FrozenBaseline(train_seed=seed); tv=c["train_validation"]
 cfg=TrainConfig(scenarios=tuple(tv["train_scenarios"]),seeds=tuple(tv["train_seeds"]),episodes=f.episodes_per_update*f.updates,steps=f.steps,rollout_episodes=f.episodes_per_update,updates=f.updates,max_nodes=f.max_nodes,ppo=f.ppo(),policy=f.policy(0),out_dir=OUT,tag=f"seed_{seed}",arm="main_baseline",reward_mode="balanced",eval_scenarios=tuple(tv["validation_scenarios"]),eval_seeds=tuple(tv["validation_seeds"]),seed=seed)
 assert_frozen(cfg,f); result=train(cfg,quiet=True)
 dump(os.path.join(OUT,f"seed_{seed}","balanced_metadata.json"),{"seed":seed,"selected_update":result["selected_update"],"checkpoint":result["policy_path"],"checkpoint_sha256":digest(result["policy_path"]),"validation":result["validation_of_selected"],"validation_without_mask":result["validation_without_mask"],"freeze_sha256":digest(contract())})
 print(json.dumps({"seed":seed,"selected_update":result["selected_update"]},ensure_ascii=False))

def report()->None:
 c=load(contract()); cfg=checked(); rows=[]
 # validation-only references; old test/test-v2 are never read.
 for method in ("rule","rolling_horizon"):
  for scene in c["train_validation"]["validation_scenarios"]:
   for env in c["train_validation"]["validation_seeds"]:
    r=evaluate_episode(method,scene,env,FrozenEvaluation()); r["training_seed"]="fixed"; rows.append(r)
 for seed in c["training_seeds"]:
  meta=load(os.path.join(OUT,f"seed_{seed}","balanced_metadata.json")); path=meta["checkpoint"]
  if digest(path)!=meta["checkpoint_sha256"]: raise RuntimeError("checkpoint changed")
  f=FrozenEvaluation(checkpoints={"ppo_baseline":path,"ppo_freshness_uncertainty":path})
  for scene in c["train_validation"]["validation_scenarios"]:
   for env in c["train_validation"]["validation_seeds"]:
    r=evaluate_episode("ppo_baseline",scene,env,f); r["training_seed"]=seed; rows.append(r)
 errors=validate_records(rows)
 if errors: raise RuntimeError("完整性失败："+";".join(errors))
 def avg(items,key): return sum(float(x[key]) for x in items)/max(1,len(items))
 summary={}
 for name in ("rule","rolling_horizon","ppo_baseline"):
  group=[x for x in rows if x["method"]==name]
  summary[name]={key:avg(group,key) for key in ("service_completion","task_timeliness","estimate_quality","resource_consumption","communication_overhead","compute_time_s","completion_rate","mean_waiting_s","n_expired","unmasked_argmax_invalid_rate") if group[0].get(key) is not None}
 report={"protocol":c,"raw_rows":rows,"summary":summary,"integrity":{"ok":True,"errors":[]},"boundaries":["Balanced is a fixed equal-weight anchor, not an overall optimum.","Compute time is evaluation-only, not reward.","No test-v3 was read or released.","References are validation-only and descriptive."]}
 dump(os.path.join(OUT,"balanced_validation_report.json"),report)
 with open(os.path.join(OUT,"balanced_validation_raw.csv"),"w",encoding="utf-8-sig",newline="") as handle:
  keys=sorted({key for row in rows for key in row if key not in ("checkpoint_extra","action_composition")}); writer=csv.DictWriter(handle,fieldnames=keys); writer.writeheader(); writer.writerows([{key:value for key,value in row.items() if key in keys} for row in rows])
 print(json.dumps({"records":len(rows),"report":os.path.join(OUT,"balanced_validation_report.json")},ensure_ascii=False))

def main(argv:Sequence[str]|None=None)->int:
 p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True); s.add_parser("prepare"); t=s.add_parser("train");t.add_argument("--seed",type=int,required=True);s.add_parser("report");a=p.parse_args(argv)
 if a.cmd=="prepare":prepare()
 elif a.cmd=="train":train_one(a.seed)
 else:report()
 return 0
if __name__=="__main__":raise SystemExit(main())
