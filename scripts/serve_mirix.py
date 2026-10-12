#!/usr/bin/env python3
"""Run the real MIRIX app with async, byte-preserving media registration.

Use a separate port from other experiments. No memory algorithm is replaced.
SDK location and database environment must be supplied by the caller.
"""
import argparse
import base64
import asyncio
import contextvars
from copy import deepcopy
import hashlib
import functools
import json
import os
from pathlib import Path
import tempfile
import time


def configure(app, api, directory, extra_body, empty_tool_confirmation=False,llm_concurrency=2,rate_limit_delay=65,llm_timeout=300):
    from fastapi import Header, HTTPException
    from mirix.services.file_manager import FileManager
    from mirix.llm_api.openai_client import OpenAIClient
    registry={}
    audits={}
    failures={}
    request_tasks={}
    gate=asyncio.Semaphore(llm_concurrency)
    scope_context=contextvars.ContextVar('br_mirix_scope',default=(None,None))
    manager=FileManager()

    async def register(messages, org, scope):
        messages=deepcopy(messages)
        for message in messages:
            if not isinstance(message.get('content'),list): continue
            for index,part in enumerate(message['content']):
                if part.get('type')!='image_url': continue
                url=part['image_url']['url']
                if not url.startswith('data:image/'):
                    raise ValueError('Validation service expects complete inline image bytes.')
                header,encoded=url.split(',',1)
                raw=base64.b64decode(encoded,validate=True)
                digest=hashlib.sha256(raw).hexdigest()
                mime=header.split(';',1)[0][5:]
                cache=registry.setdefault(scope,{})
                if digest not in cache:
                    path=directory/f'{scope}_{digest}.{mime.split("/")[1]}'
                    path.write_bytes(raw)
                    record=await manager.create_file_metadata_from_path(str(path),org)
                    cache[digest]=(record.id,path)
                message['content'][index]={'type':'database_image_id','image_id':cache[digest][0]}
        return messages

    # Keep the original FastAPI dependency graph and SDK handler; only register
    # media asynchronously before the handler's synchronous message conversion.
    def leaf_routes(router):
        for route in router.routes:
            if hasattr(route,'original_router'):
                yield from leaf_routes(route.original_router)
            else:
                yield route
    configured_routes=[]
    for route in leaf_routes(app.router):
        if getattr(route,'path',None) not in {'/memory/add','/memory/add_sync','/memory/retrieve/conversation'}:
            continue
        original=route.dependant.call
        def wrap(handler):
            @functools.wraps(handler)
            async def call(**kwargs):
                request=kwargs['request']
                _,org=await api.get_client_and_org(kwargs.get('x_client_id'),kwargs.get('x_org_id'))
                scope=kwargs.get('x_client_id')
                request.messages=await register(request.messages,org,scope)
                token=scope_context.set((scope,(request.filter_tags or {}).get('source_session') if hasattr(request,'filter_tags') else None))
                failure=asyncio.get_running_loop().create_future()
                failures[scope]=failure
                task=asyncio.create_task(handler(**kwargs))
                try:
                    await asyncio.wait([task,failure],return_when=asyncio.FIRST_COMPLETED)
                    if failure.done():
                        raise HTTPException(status_code=502,detail='Memory model failed: '+failure.result())
                    return await task
                finally:
                    pending={task,*request_tasks.get(scope,set())}
                    for owned in pending:
                        if not owned.done():owned.cancel()
                    await asyncio.gather(*pending,return_exceptions=True)
                    failures.pop(scope,None)
                    scope_context.reset(token)
            return call
        wrapped=wrap(original)
        route.dependant.call=wrapped
        route.endpoint=wrapped
        configured_routes.append(route.path)

    original_topics=api.extract_topics_and_temporal_info
    async def topics(messages,llm_config):
        if messages and isinstance(messages[0],dict) and 'role' in messages[0]:
            messages=api.openai_messages_to_content_parts(messages,include_non_text=True)
        return await original_topics(messages,llm_config)
    api.extract_topics_and_temporal_info=topics

    if extra_body or empty_tool_confirmation:
        original_build=OpenAIClient.build_request_data
        async def build(self,*args,**kwargs):
            data=await original_build(self,*args,**kwargs)
            data['extra_body']={**data.get('extra_body',{}),**extra_body}
            if empty_tool_confirmation:
                for tool in data.get('tools') or []:
                    function=tool.get('function',{})
                    params=function.get('parameters',{})
                    if function.get('name')=='finish_memory_update' and not params.get('properties'):
                        function['parameters']={'type':'object','properties':{'acknowledged':{'type':'boolean','enum':[True]}},
                            'required':['acknowledged'],'additionalProperties':False}
            return data
        OpenAIClient.build_request_data=build

    original_request=OpenAIClient.request
    async def request(self,data):
        scope,session=scope_context.get()
        adapted_tools={t['function']['name'] for t in data.get('tools') or []
                       if set(t['function'].get('parameters',{}).get('properties',{}))=={'acknowledged'}} if empty_tool_confirmation else set()
        audit={'model':data.get('model'),'source_session':session,'started':time.time(),
               'request_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),
               'image_parts':sum(p.get('type')=='image_url' for m in data.get('messages',[])
                   if isinstance(m.get('content'),list) for p in m['content'])}
        audits.setdefault(scope,[]).append(audit)
        task=asyncio.current_task()
        request_tasks.setdefault(scope,set()).add(task)
        try:
            async with gate:
                for attempt in range(6):
                    try:
                        response=await asyncio.wait_for(original_request(self,data),timeout=llm_timeout)
                        break
                    except Exception as error:
                        if type(error).__name__ not in {'LLMRateLimitError','RateLimitError'} or attempt==5:
                            raise
                        audit['rate_limit_retries']=attempt+1
                        await asyncio.sleep(rate_limit_delay)
            if empty_tool_confirmation:
                for choice in response.get('choices',[]):
                    for tool in choice.get('message',{}).get('tool_calls') or []:
                        function=tool['function']
                        if function.get('name') in adapted_tools:
                            if json.loads(function['arguments'])!={'acknowledged':True}:
                                raise ValueError('Invalid empty-tool completion acknowledgement.')
                            function['arguments']='{}'
            audit.update(finish_reason=response['choices'][0].get('finish_reason'),usage=response.get('usage',{}),completed=True)
            return response
        except Exception as error:
            audit.update(error_type=type(error).__name__,completed=True)
            failure=failures.get(scope)
            if failure is not None and not failure.done():failure.set_result(type(error).__name__)
            raise
        except asyncio.CancelledError:
            audit.update(error_type='CancelledError',completed=True)
            raise
        finally:
            audit['elapsed_seconds']=round(time.time()-audit['started'],3)
            request_tasks[scope].discard(task)
    OpenAIClient.request=request

    async def delete_media(scope, x_client_id: str = Header(None)):
        if scope != x_client_id:
            raise HTTPException(status_code=403,detail='Media cleanup requires the owning question client.')
        records=registry.pop(scope,{})
        for identifier,path in records.values():
            await manager.delete_file_metadata(identifier)
            path.unlink(missing_ok=True)
        return {'success':True,'deleted_media':len(records)}
    app.add_api_route('/evaluation/media/{scope}',delete_media,methods=['DELETE'])
    async def audit(scope,x_client_id: str = Header(None)):
        if scope != x_client_id: raise HTTPException(status_code=403,detail='Question client required.')
        return {'calls':audits.get(scope,[])}
    app.add_api_route('/evaluation/audit/{scope}',audit,methods=['GET'])
    async def tasks():
        output=[]
        for task in asyncio.all_tasks():
            coroutine=task.get_coro();stack=[]
            for _ in range(15):
                frame=getattr(coroutine,'cr_frame',None) or getattr(coroutine,'gi_frame',None)
                if frame:stack.append({'function':frame.f_code.co_name,'file':Path(frame.f_code.co_filename).name,'line':frame.f_lineno})
                coroutine=getattr(coroutine,'cr_await',None) or getattr(coroutine,'gi_yieldfrom',None)
                if coroutine is None:break
            output.append({'name':task.get_name(),'stack':stack})
        return output
    app.add_api_route('/evaluation/debug/tasks',tasks,methods=['GET'])
    async def protocol():
        return {'media_registration_protocol':'async_inline_bytes_v1','llm_extra_body':extra_body,
                'query_media_preserved':'/memory/retrieve/conversation' in configured_routes,
                'configured_routes':configured_routes,'empty_tool_confirmation':empty_tool_confirmation,
                'llm_concurrency':llm_concurrency,'rate_limit_retry_delay':rate_limit_delay,
                'llm_request_timeout':llm_timeout,'fail_fast_model_errors':True}
    app.add_api_route('/evaluation/protocol',protocol,methods=['GET'])
    app.state.evaluation_media_protocol='async_inline_bytes_v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=8532)
    p.add_argument('--llm-extra-body',default='{}')
    p.add_argument('--state-dir',type=Path,required=True,help='Private SQLite/config directory; must be separate from other MIRIX services.')
    p.add_argument('--empty-tool-confirmation',action='store_true',help='Explicit workaround for providers that stall on zero-argument tool schemas.')
    p.add_argument('--llm-concurrency',type=int,default=2)
    p.add_argument('--rate-limit-retry-delay',type=float,default=65)
    p.add_argument('--llm-request-timeout',type=float,default=300)
    args=p.parse_args()
    extra=json.loads(args.llm_extra_body)
    if not isinstance(extra,dict) or set(extra)-{'enable_thinking','reasoning_effort'}:
        raise ValueError('Only explicit provider thinking controls are supported.')
    if args.llm_concurrency<1 or args.rate_limit_retry_delay<0 or args.llm_request_timeout<=0:
        raise ValueError('Invalid service concurrency or retry delay.')
    state=args.state_dir.resolve()
    state.mkdir(parents=True,exist_ok=True)
    config=state/'config.ini'
    config.write_text(''.join(f'[{name}_storage]\ntype = sqlite\npath = {state}\n\n' for name in ['recall','archival','metadata']))
    os.environ.update(MEMGPT_CONFIG_PATH=str(config),MIRIX_REDIS_ENABLED='false',MIRIX_PG_URI='')
    for name in ['MIRIX_PG_USER','MIRIX_PG_PASSWORD','MIRIX_PG_HOST','MIRIX_PG_DB','MIRIX_PG_PORT']:
        os.environ.pop(name,None)
    from mirix.server import rest_api as api
    import uvicorn
    with tempfile.TemporaryDirectory(prefix='br_mirix_media_') as temporary:
        configure(api.app,api,Path(temporary),extra,args.empty_tool_confirmation,args.llm_concurrency,args.rate_limit_retry_delay,args.llm_request_timeout)
        uvicorn.run(api.app,host='127.0.0.1',port=args.port,log_level='warning')

if __name__=='__main__': main()
