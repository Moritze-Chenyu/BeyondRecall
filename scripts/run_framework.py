#!/usr/bin/env python3
"""Run framework inference with isolated question state and optional subset selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait
import importlib.util
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import uuid

PACKAGE = Path(__file__).resolve().parents[1]
ASSETS = PACKAGE / 'runtime_assets'
sys.path.insert(0, str(PACKAGE/'src'))
from beyond_recall_eval.configuration import resolved_profile
from beyond_recall_eval.dataset import BeyondRecallDataset
from beyond_recall_eval.manifest import build_inference_manifest, hash_file, write_or_validate_manifest
from beyond_recall_eval.runner import run_inference, latest_by_question
from beyond_recall_eval.sampling import select_trajectories
from beyond_recall_eval.systems import create_system

STATE = {}


def sdk_revision(root, package):
    """Identify a checkout or verify a bundled source snapshot without .git."""
    metadata = root / 'SDK_SOURCE.json'
    if metadata.exists():
        source = json.loads(metadata.read_text())
        actual = {str(p.relative_to(root)): hash_file(p)
                  for p in sorted((root / package).rglob('*.py'))}
        if actual != source['python_source_sha256']:
            raise ValueError('Bundled SDK source differs from SDK_SOURCE.json.')
        return source['commit']
    return subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()


def observed_create(info, config):
    system = create_system(info, config)
    path = STATE['output']/'events'/f'{info.namespace}.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    def event(value):
        with path.open('a') as f:
            f.write(json.dumps(value)+'\n')
            f.flush()
    event({'event':'initialized','namespace':info.namespace,'runtime':system.runtime_info()})
    ingest, close = system.ingest, system.close
    def audited_ingest(session):
        event({'event':'ingest_start','source_session':session.index,'date':session.date,
               'text_sha256':__import__('hashlib').sha256(session.text.encode()).hexdigest(),
               'source_units':list(session.source_units),'media_assets':[a.asset_id for a in session.media]})
        ingest(session)
        event({'event':'ingest_complete','source_session':session.index})
    def audited_close():
        result=close()
        event({'event':'closed',**result})
        return result
    system.ingest, system.close = audited_ingest,audited_close
    return system


def worker(index, keys):
    return run_inference(dataset_root=STATE['dataset'], trajectories=STATE['trajectories'],
        config=STATE['config'], profile_name=STATE['name'], profile=STATE['profile'],
        create_system=observed_create, output_path=STATE['output']/f'worker_{index}.jsonl',
        run_manifest=STATE['manifest'], framework_version=STATE['version'],
        caption_store_path=STATE['captions'], question_keys=set(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--profile',choices=['mem0_agentic','mem0_oss','amem','mirix','rag_dense','rag_hybrid','rag_bm25_caption'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--sample-manifest',type=Path,help='Optional frozen question selection; omit for the full dataset.')
    parser.add_argument('--caption-store',type=Path,default=ASSETS/'captions.json')
    parser.add_argument('--previous-config',type=Path,default=PACKAGE/'examples/api_config.py')
    parser.add_argument('--framework-source',type=Path)
    parser.add_argument('--video-segment-manifest',type=Path,
        default=ASSETS/'video_segments/manifest.json')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--mirix-memory-model',default='Qwen/Qwen3.5-27B',help='Memory model used by MIRIX; retained in run configuration.')
    args = parser.parse_args()
    if not args.framework_source and args.profile in {'amem','mirix','mem0_oss','mem0_agentic'}:
        bundled=PACKAGE/'frameworks'/{'amem':'A-mem','mirix':'MIRIX','mem0_oss':'Mem0','mem0_agentic':'Mem0-Agentic'}[args.profile]
        if bundled.exists():args.framework_source=bundled
    if not 1<=args.workers<=4:
        raise ValueError('Use 1–4 isolated framework workers.')
    for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
        if os.environ.get(key,'').lower().startswith('socks5h://'):
            os.environ.pop(key,None)
    os.environ.update(MEM0_TELEMETRY='false',HF_HUB_DISABLE_XET='1',ANONYMIZED_TELEMETRY='False',
                      OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    if args.framework_source:
        sys.path.insert(0,str(args.framework_source.resolve()))
    sys.path.insert(0,str(args.previous_config.resolve().parent))
    spec = importlib.util.spec_from_file_location('private_previous',args.previous_config)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    os.environ.update(BR_API_KEY=previous.ANSWER_API_KEY,BR_API_BASE_URL=previous.ANSWER_BASE_URL,
                      BR_FRAME_MEMORY_KEY=previous.MEM0_API_KEY,BR_FRAME_MEMORY_URL=previous.MEM0_BASE_URL)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    trajectories = select_trajectories(list(BeyondRecallDataset(args.dataset).trajectories()),args.sample_manifest,args.dataset)
    keys = [f'{t.key}/q{q.index}' for t in trajectories for q in t.questions]
    config = json.loads((PACKAGE/'config/api_pdf_cropped_10.json').read_text())
    answer = config['models']['answer']
    answer.update(model=previous.ANSWER_MODEL,revision='previous-api-unversioned',revision_env='',
        chat_template_revision='previous-api-unversioned',extra_body={'enable_thinking':False},
        timeout_seconds=180,max_tokens=1024)
    memory = config['models']['memory_extraction_and_evolution']
    memory.update(backend='api',model=previous.MEM0_MODEL,revision='previous-api-unversioned',revision_env='',
        chat_template_revision='previous-api-unversioned',api_key_env='BR_FRAME_MEMORY_KEY',
        base_url_env='BR_FRAME_MEMORY_URL',max_tokens=2000,timeout_seconds=180,
        extra_body={'enable_thinking':False},require_verified_processing=False)
    config['models']['text_embedding'].update(model=previous.MEM0_EVAL_EMBED_MODEL,dimension=384,
        revision='fastembed-cached-model',revision_env='')
    quality = config['media_processing']['quality']
    quality.update(pdf_mode='legacy_caption',caption_policy='legacy_frozen')
    profile = config['method_profiles'][args.profile]
    profile['session_chunk_size']=1
    version = 'beyond-recall-framework-v1'
    if args.profile in {'mem0_oss','mem0_agentic'}:
        metadata=json.loads((args.framework_source/'SDK_SOURCE.json').read_text())
        version=args.profile+'-'+metadata['version']+'-'+sdk_revision(args.framework_source,'mem0')
        profile['sdk_source_root']=str(args.framework_source.resolve())
        profile['framework_config']={'llm':{'provider':'openai','config':{
            'model':memory['model'],'api_key_env':'BR_FRAME_MEMORY_KEY',
            'openai_base_url_env':'BR_FRAME_MEMORY_SDK_URL','temperature':0.1,'top_p':0.1,'max_tokens':2000}},
            'embedder':{'provider':'fastembed','config':{'model':previous.MEM0_EVAL_EMBED_MODEL,'embedding_dims':384}},
            'vector_store':{'provider':'qdrant','config':{'embedding_model_dims':384,'on_disk':True}}}
        os.environ['BR_FRAME_MEMORY_SDK_URL']=previous.MEM0_BASE_URL.rstrip('/').removesuffix('/chat/completions')
    elif args.profile=='amem':
        if not args.framework_source:
            raise ValueError('A-Mem requires --framework-source pointing to its real SDK checkout.')
        profile.update(implementation='amem_caption',retrieval_top_k=8,
            framework_config={'embedding_model':'all-MiniLM-L6-v2','metadata_mode':'simple','evolution_mode':'llm'})
        memory.update(model=previous.ANSWER_MODEL,max_tokens=4096)
        os.environ.update(BR_FRAME_MEMORY_KEY=previous.ANSWER_API_KEY,BR_FRAME_MEMORY_URL=previous.ANSWER_BASE_URL)
        version='A-Mem-'+sdk_revision(args.framework_source,'agentic_memory')
    elif args.profile.startswith('rag_'):
        profile.update(retrieval_top_k=8)
        if args.profile!='rag_bm25_caption':
            profile['implementation']='dense_rag'
            profile['embedding_role']='text_embedding'
            version='fastembed-'+importlib.metadata.version('fastembed')
    elif args.profile=='mirix':
        if not args.framework_source:
            raise ValueError('MIRIX requires --framework-source pointing to its SDK checkout.')
        os.environ.update(BR_API_KEY=previous.FULL_CONTEXT_MULTIMODAL_API_KEY,
            BR_API_BASE_URL=previous.FULL_CONTEXT_MULTIMODAL_BASE_URL,
            BR_FRAME_MEMORY_KEY=previous.FULL_CONTEXT_MULTIMODAL_API_KEY,
            BR_FRAME_MEMORY_URL=previous.FULL_CONTEXT_MULTIMODAL_BASE_URL,
            BR_MIRIX_URL=os.getenv('MIRIX_BASE_URL','http://localhost:8531'),
            BR_MIRIX_LLM_URL=previous.FULL_CONTEXT_MULTIMODAL_BASE_URL.rstrip('/').removesuffix('/chat/completions'))
        answer.update(model=previous.FULL_CONTEXT_MULTIMODAL_MODEL,pdf_transport='all_pages_images',pdf_render_dpi=300)
        memory.update(model=args.mirix_memory_model,visual_transport='all_frames_lossless_tiles')
        llm={'model':memory['model'],'model_endpoint_type':'openai','api_key_env':'BR_FRAME_MEMORY_KEY',
             'model_endpoint_env':'BR_MIRIX_LLM_URL','context_window':262144,'temperature':0}
        profile.update(implementation='mirix_native',retrieval_top_k=120,framework_config={
            'retrieval_limit_per_type':10,
            'base_url_env':'BR_MIRIX_URL','timeout_seconds':1800,'org_id':'beyond-recall',
            'agent_config':{'llm_config':llm,'topic_extraction_llm_config':llm,
              'embedding_config':{'embedding_model':previous.MEM0_EVAL_EMBED_MODEL,
                  'embedding_endpoint_type':'fastembed','embedding_dim':384},
              'build_embeddings_for_memory':True,'meta_agent_config':{
                'agents':[{'core_memory_agent':{'blocks':[{'label':'human','value':''},
                          {'label':'persona','value':'I am a helpful assistant.'}]}},
                          'episodic_memory_agent','semantic_memory_agent','procedural_memory_agent',
                          'resource_memory_agent','knowledge_vault_memory_agent','meta_memory_agent']}}})
        video_manifest=args.video_segment_manifest
        if os.getenv('BR_MIRIX_MEDIA_REGISTRATION_PROTOCOL'):
            profile['framework_config'].update(media_registration_protocol=os.environ['BR_MIRIX_MEDIA_REGISTRATION_PROTOCOL'],
                llm_extra_body=memory['extra_body'],empty_tool_confirmation=os.getenv('BR_MIRIX_EMPTY_TOOL_CONFIRMATION')=='1')
        quality.update(pdf_mode='fixed_cropped_pdf',video_mode='segmented_native_transcript',video_transport='video_url',
            video_segment_manifest=str(video_manifest),video_segment_manifest_sha256=hash_file(video_manifest))
        version='MIRIX-'+sdk_revision(args.framework_source,'mirix')
    resolved=resolved_profile(config,args.profile)
    config_path=args.output_dir/'configuration.json'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    lock=args.output_dir/'framework.lock.json'
    sdk_files={}
    if args.framework_source:
        sdk_root=args.framework_source/{'amem':'agentic_memory','mirix':'mirix','mem0_oss':'mem0','mem0_agentic':'mem0'}[args.profile]
        sdk_files={str(p.relative_to(args.framework_source)):hash_file(p) for p in sorted(sdk_root.rglob('*.py'))}
    lock.write_text(json.dumps({'framework_version':version,'python':sys.version,
        'sdk_source_sha256':sdk_files,
        'service_wrapper_sha256':hash_file(PACKAGE/'scripts/serve_mirix.py') if args.profile=='mirix' and os.getenv('BR_MIRIX_MEDIA_REGISTRATION_PROTOCOL') else None,
        'dependencies':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},indent=2)+'\n')
    manifest_path=args.output_dir/'predictions.run.json'
    run_id=json.loads(manifest_path.read_text())['run_id'] if manifest_path.exists() else 'framework-'+args.profile+'-'+uuid.uuid4().hex
    manifest=build_inference_manifest(package_root=PACKAGE,dataset_root=args.dataset,config=config,
        profile_name=args.profile,resolved_profile=resolved,adapter_file=PACKAGE/'src/beyond_recall_eval/frameworks.py',
        framework_lock=lock,caption_store=args.caption_store if resolved['input_mode']=='caption' else None,
        evidence_manifest=None,framework_version=version,run_id=run_id,
        sample_manifest=args.sample_manifest)
    write_or_validate_manifest(args.output_dir/'predictions.run.json',manifest)
    STATE.update(dataset=args.dataset,trajectories=trajectories,config=config,name=args.profile,profile=resolved,
        output=args.output_dir,version=version,manifest=manifest,captions=args.caption_store if resolved['input_mode']=='caption' else None)
    if not keys:
        raise ValueError('No questions selected for inference.')
    workers=min(args.workers,len(keys))
    print(json.dumps({'profile':args.profile,'framework_version':version,'questions':len(keys),
        'answer_model':answer['model'],'memory_model':memory['model'],'workers':workers}),flush=True)
    context=multiprocessing.get_context('fork')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        pending={pool.submit(worker,i,keys[i::workers]) for i in range(workers)}
        while pending:
            done,pending=wait(pending,timeout=20)
            for future in done:
                future.result()
            rows={}
            for i in range(workers):rows.update(latest_by_question(args.output_dir/f'worker_{i}.jsonl'))
            print(json.dumps({'completed':len(rows),'successful':sum(not r['error'] for r in rows.values()),
                              'errors':sum(bool(r['error']) for r in rows.values())}),flush=True)
    rows={}
    for i in range(workers):rows.update(latest_by_question(args.output_dir/f'worker_{i}.jsonl'))
    (args.output_dir/'predictions.jsonl').write_text(''.join(json.dumps(rows[k],ensure_ascii=False)+'\n' for k in sorted(rows)))
    report={'profile':args.profile,'expected_questions':len(keys),'completed_questions':len(rows),
        'successful_inference':sum(not r['error'] for r in rows.values()),'inference_errors':sum(bool(r['error']) for r in rows.values()),
        'framework_version':version,'experiment_kind':'framework inference; answer judging is a separate stage',
        'errors':{k:r['error'] for k,r in rows.items() if r['error']}}
    (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
