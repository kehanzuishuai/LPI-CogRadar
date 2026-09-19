"""冻结偏好集的单模型 Preference-Conditioned PPO 机制验证。"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from typing import Any,Dict,Sequence
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)));sys.path.insert(0,ROOT)
from resource_management.contract_v1 import verify_frozen,contract_digest
from rl_resource.ablation import FrozenBaseline,assert_frozen
from rl_resource.policy import ActorCritic
from rl_resource.train import TrainConfig,train,evaluate
CFG=os.path.join(ROOT,"config","preference_ppo_v1.json"); SHA=CFG.replace(".json",".sha256"); BAL=os.path.join(ROOT,"config","balanced_protocol_v1.json"); OUT=os.path.join(ROOT,"output","rl_resource","preference_ppo")
def dig(p):
 h=hashlib.sha256();h.update(open(p,"rb").read());return h.hexdigest()
def load(p):return json.load(open(p,encoding="utf-8"))
def dump(p,x):os.makedirs(os.path.dirname(p),exist_ok=True);json.dump(x,open(p,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
def freeze():return os.path.join(OUT,"training_freeze.json")
def checked():
 c=load(CFG)
 if dig(CFG)!=open(SHA).read().strip() or not verify_frozen(strict=False)["ok"]:raise RuntimeError("协议或资源契约未冻结")
 b=load(BAL)
 if not b["test_v3_sealed"]:raise RuntimeError("test-v3 必须封存")
 for name,p in c["preferences"].items():
  if len(p)!=5 or any(x<0 for x in p) or abs(sum(p)-1)>1e-9:raise ValueError(name+" 非 simplex")
 return c,b
def prepare():
 c,b=checked();dump(freeze(),{"protocol_sha256":dig(CFG),"balanced_protocol_sha256":dig(BAL),"contract":contract_digest(),"preferences":c["preferences"],"sampling":c["preference_sampling"],"seeds":c["training_seeds"],"test_v3":"sealed"});print(freeze())
def train_one(seed):
 fz=load(freeze());c=load(CFG)
 if seed not in fz["seeds"]:raise ValueError("seed 未冻结")
 f=FrozenBaseline(train_seed=seed); prefs=tuple(tuple(x) for x in fz["preferences"].values())
 cfg=TrainConfig(scenarios=f.train_scenarios,seeds=f.train_seeds,episodes=f.episodes_per_update*f.updates,steps=f.steps,rollout_episodes=f.episodes_per_update,updates=f.updates,max_nodes=f.max_nodes,ppo=f.ppo(),policy=f.policy(0),out_dir=OUT,tag=f"seed_{seed}",arm="main_baseline",reward_mode="balanced",preference_conditioned=True,preference_set=prefs,evaluation_preference=tuple(fz["preferences"]["balanced"]),eval_scenarios=("rm_validation_node_outage","rm_validation_handover","rm_validation_sensor_bias"),eval_seeds=(211,223,227),seed=seed)
 assert_frozen(cfg,f);s=train(cfg,quiet=True);dump(os.path.join(OUT,f"seed_{seed}","metadata.json"),{"seed":seed,"checkpoint":s["policy_path"],"sha256":dig(s["policy_path"]),"selected_update":s["selected_update"],"freeze_sha256":dig(freeze())});print(seed)
def report():
 fz=load(freeze());prefs=fz["preferences"];rows=[]
 for seed in fz["seeds"]:
  m=load(os.path.join(OUT,f"seed_{seed}","metadata.json"));
  if dig(m["checkpoint"])!=m["sha256"]:raise RuntimeError("checkpoint changed")
  model,_=ActorCritic.load(m["checkpoint"],map_location="cpu")
  for name,p in prefs.items():
   f=FrozenBaseline(train_seed=seed);cfg=TrainConfig(scenarios=f.train_scenarios,seeds=f.train_seeds,steps=f.steps,max_nodes=f.max_nodes,ppo=f.ppo(),policy=f.policy(0),arm="main_baseline",reward_mode="balanced",preference_conditioned=True,evaluation_preference=tuple(p),eval_scenarios=("rm_validation_node_outage","rm_validation_handover","rm_validation_sensor_bias"),eval_seeds=(211,223,227),seed=seed)
   e=evaluate(model,cfg,__import__('torch').device('cpu'),cfg.eval_scenarios,cfg.eval_seeds);rows += [{"training_seed":seed,"preference":name,**r} for r in e["rows"]]
 def mean(rs,k):return sum(float(x[k]) for x in rs)/len(rs)
 table={}
 for n in prefs:
  rs=[r for r in rows if r["preference"]==n];actions={a:sum(r["action_composition"][a] for r in rs) for a in ("idle","sample","process","share")};table[n]={k:mean(rs,k) for k in ("completion_rate","timeliness","estimate_quality","resource_consumption","comm_overhead_bytes","mean_waiting_s","n_expired","illegal_action_rate","illegal_probability_mass")};table[n]["actions"]=actions
 gate={"completion_vs_resource":table["completion"]["completion_rate"]>table["resource_saving"]["completion_rate"],"resource_saving":table["resource_saving"]["resource_consumption"]<table["completion"]["resource_consumption"],"communication_saving":table["communication_saving"]["comm_overhead_bytes"]<table["completion"]["comm_overhead_bytes"]}
 gate["passed"]=gate["completion_vs_resource"] and gate["resource_saving"] and gate["communication_saving"]
 def dominates(a,b):
  ge=(a["completion_rate"]>=b["completion_rate"] and a["timeliness"]>=b["timeliness"] and a["estimate_quality"]>=b["estimate_quality"] and a["resource_consumption"]<=b["resource_consumption"] and a["comm_overhead_bytes"]<=b["comm_overhead_bytes"])
  strict=(a["completion_rate"]>b["completion_rate"] or a["timeliness"]>b["timeliness"] or a["estimate_quality"]>b["estimate_quality"] or a["resource_consumption"]<b["resource_consumption"] or a["comm_overhead_bytes"]<b["comm_overhead_bytes"])
  return ge and strict
 pareto=[n for n,a in table.items() if not any(dominates(b,a) for m,b in table.items() if m!=n)]
 result={"freeze":fz,"rows":rows,"preference_table":table,"validation_pareto_nondominated":pareto,"mechanism_gate":gate,"test_v3":"sealed; not read or released","boundaries":["No single score/ranking.","Balanced fixed failure is retained.","Communication preference failed to affect share/communication, so mechanism gate fails and test-v3 remains sealed."]};dump(os.path.join(OUT,"validation_report.json"),result)
 with open(os.path.join(OUT,"validation_raw.csv"),"w",encoding="utf-8-sig",newline="") as h:
  keys=["training_seed","preference","scenario","seed","completion_rate","timeliness","estimate_quality","resource_consumption","comm_overhead_bytes","mean_waiting_s","n_expired","illegal_action_rate","illegal_probability_mass"];w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows([{k:r.get(k) for k in keys} for r in rows])
 print(json.dumps({"rows":len(rows),"gate":gate},ensure_ascii=False))
def main(a:Sequence[str]|None=None):
 p=argparse.ArgumentParser();s=p.add_subparsers(dest="x",required=True);s.add_parser("prepare");t=s.add_parser("train");t.add_argument("--seed",type=int,required=True);s.add_parser("report");q=p.parse_args(a); {"prepare":prepare,"train":lambda:train_one(q.seed),"report":report}[q.x]();return 0
if __name__=="__main__":raise SystemExit(main())
