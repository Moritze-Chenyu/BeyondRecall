#!/usr/bin/env python3
"""Exercise distinct real Mem0 SDKs with deterministic test transport; no API calls."""
import os,sys,json,ast,re,hashlib,argparse,subprocess
from pathlib import Path
from types import SimpleNamespace
os.environ['MEM0_TELEMETRY']='false';sys.dont_write_bytecode=True
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
 if os.environ.get(k,'').startswith('socks5h://'):os.environ.pop(k)
r=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile',choices=['all','mem0_agentic','mem0_oss'],default='all')
parser.add_argument('--dataset',type=Path)
parser.add_argument('--sample-manifest',type=Path)
parser.add_argument('--caption-store',type=Path)
parser.add_argument('--output-dir',type=Path,default=r/'results/mem0_sdk_checks')
args=parser.parse_args()
if any([args.dataset,args.sample_manifest,args.caption_store]) and not all([args.dataset,args.sample_manifest,args.caption_store]):
 parser.error('Dataset checks require --dataset, --sample-manifest and --caption-store together.')
if args.profile=='all':
 for profile_name in ['mem0_agentic','mem0_oss']:
  command=[sys.executable,str(Path(__file__).resolve()),'--profile',profile_name,'--output-dir',str(args.output_dir)]
  if args.dataset:command+=['--dataset',str(args.dataset),'--sample-manifest',str(args.sample_manifest),'--caption-store',str(args.caption_store)]
  subprocess.run(command,check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
 raise SystemExit(0)
sys.path.insert(0,str(r/'src'))
from beyond_recall_eval.frameworks import load_mem0_sdk,Mem0AgenticSystem,Mem0OSSSystem
from beyond_recall_eval.configuration import resolved_profile
from beyond_recall_eval.dataset import BeyondRecallDataset
from beyond_recall_eval.sampling import select_trajectories
from beyond_recall_eval.temporal import visible_history
from beyond_recall_eval.rendering import InputBuilder,CaptionStore
from beyond_recall_eval.models import TrajectoryInfo,PreparedSession,PreparedQuestion
name=args.profile;variant='legacy_non_graph' if name=='mem0_agentic' else 'oss_mainline'
Memory,meta=load_mem0_sdk(variant)
from mem0.utils.factory import EmbedderFactory,LlmFactory
class Embedding:
 config=SimpleNamespace(model='deterministic-test-embedding')
 def embed(self,text,*args,**kwargs):return [1.0]+[0.0]*383
 def embed_batch(self,texts,*args,**kwargs):return [self.embed(t) for t in texts]
EmbedderFactory.create=staticmethod(lambda *a,**k:Embedding())
original_llm=LlmFactory.create;transport_calls=[]
def llm_factory(*args,**kwargs):
 llm=original_llm(*args,**kwargs)
 def complete(**kw):
  prompt='\n'.join(m['content'] for m in kw['messages']);transport_calls.append(len(prompt))
  fact='The SDK boundary check fixture is active.'
  if 'validation-blue' in prompt or 'The preferred color is blue.' in prompt:fact='The preferred color is blue.'
  elif 'validation-red' in prompt or 'The preferred color is red.' in prompt:fact='The preferred color is red.'
  if variant=='oss_mainline':data={'memory':[{'text':fact}]}
  elif 'The new retrieved facts' in prompt:
   marker='Below is the current content of my memory which I have collected till now.'
   if marker in prompt:
    block=prompt.rsplit(marker,1)[1].split('The new retrieved facts',1)[0];old=ast.literal_eval(block.split('```',2)[1].strip())
   else:old=[]
   if old:data={'memory':[{'id':str(old[0]['id']),'event':'UPDATE','text':fact}]}
   else:data={'memory':[{'event':'ADD','text':fact}]}
  else:data={'facts':[fact]}
  return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(data)))],usage=None)
 llm.client.chat.completions.create=complete
 return llm
LlmFactory.create=staticmethod(llm_factory)
class Answer:
 last_audit={'test_only':True}
 def generate(self,parts,prompt):return 'SDK boundary test; no model accuracy claim.'
c=json.loads((r/'config/api_quality.json').read_text());q=resolved_profile(c,name)
os.environ.update(BR_TEST_KEY='dummy-sdk-test-key',BR_FRAME_MEMORY_SDK_URL='http://127.0.0.1:1/v1')
q['framework_config']['llm']['config']['api_key_env']='BR_TEST_KEY'
q['framework_config']['embedder']['config']['model']='deterministic-test-embedding'
model=c['models']['memory_extraction_and_evolution'];q['framework_config']['llm']['config']['model']=model['model']
config={'profile':q,'models':c['models'],'runtime_expectations':{},'prompts':c['prompts'],'media_processing':c['media_processing']}
cls=Mem0AgenticSystem if name=='mem0_agentic' else Mem0OSSSystem
# Actual SDK update/additive semantics and source lineage.
info=TrajectoryInfo('check','fixture','br_fixture_'+name,('A','B'),'A',0)
system=cls(info,config,Answer());state=Path(system.state.name)
system.ingest(PreparedSession(0,'2026-01-01T00:00:00',(),'validation-red',(),('source-red',)))
system.ingest(PreparedSession(1,'2026-01-02T00:00:00',(),'validation-blue',(),('source-blue',)))
memories=system._all();result=system.answer(PreparedQuestion(0,'What is the preferred color?',(),()))
assert result.retrieval_trace
if name=='mem0_agentic':
 assert len(memories)==1,memories
 assert set(memories[0]['metadata']['source_units'])=={'source-red','source-blue'}
 assert memories[0]['memory']=='The preferred color is blue.'
else:assert len(memories)==2,memories
system.close();assert not state.exists()
try:load_mem0_sdk('oss_mainline' if variant=='legacy_non_graph' else 'legacy_non_graph')
except ValueError as e:assert 'separate processes' in str(e)
else:raise AssertionError('SDK cross-loading should be rejected')
# Optionally use selected real benchmark inputs and matching frozen captions.
root=args.dataset
trajectories=select_trajectories(list(BeyondRecallDataset(root).trajectories()),args.sample_manifest,root) if root else []
store=CaptionStore(args.caption_store)
rows=[];visible_count=0;future_count=0
for t in trajectories:
 for question in t.questions:
  visible=visible_history(t.sessions,question);future_count+=len(t.sessions)-len(visible);visible_count+=len(visible)
  builder=InputBuilder('caption',store,caption_error='fail',media_quality={**c['media_processing']['quality'],'pdf_mode':'legacy_caption','caption_policy':'legacy_frozen'})
  sessions=[builder.prepare_session(s) for s in visible];prepared=builder.prepare_question(question)
  info=TrajectoryInfo('check',hashlib.sha256(t.key.encode()).hexdigest()[:16],'br_check_'+name+'_'+str(len(rows)),t.persons,t.persons[0],question.index)
  system=cls(info,config,Answer());state=Path(system.state.name)
  try:
   for session in sessions:system.ingest(session)
   result=system.answer(prepared);assert result.retrieval_trace
   assert result.diagnostics['mem0_variant']==variant
   assert not system.failures
   rows.append({'question_key':f'{t.key}/q{question.index}','visible_sessions':len(sessions),'timestamped':bool(question.timestamp),'memory_model_calls':len(system.calls),'retrieved_memories':len(result.retrieval_trace)})
  finally:system.close()
  assert not state.exists()
if root:
 assert rows, 'The supplied sample selection is empty.'
report={'profile':name,'SDK_version':meta['version'],'real_SDK_and_Qdrant':True,'LLM_transport_and_embeddings':'scripted deterministic fixtures; no real API calls or accuracy claims','status':'passed','fixture_checks_passed':True,'benchmark_questions_passed':len(rows),'visible_sessions':visible_count,'future_sessions_excluded':future_count,'SDK_algorithm_difference_checked':True,'source_lineage_checked':True,'SDK_cross_loading_rejected':True,'namespace_cleanup_passed':True,'rows':rows}
args.output_dir.mkdir(parents=True,exist_ok=True)
(args.output_dir/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
