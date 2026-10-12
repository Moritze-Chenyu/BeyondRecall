# Method implementation status

A configuration profile describes an evaluation protocol. Its presence does not
mean that this repository implements the named method or reproduces paper results.
The `implementation_status` field in every shipped configuration makes this explicit.

| Method / Baseline | CLI Profile | Status | Runtime implementation and requirements |
|---|---|---|---|
| Full-Context Caption | `full_context_caption` | Built in | Complete visible history with frozen captions; general CLI |
| Full-Context Multimodal | `full_context_multimodal` | Built in | Complete visible history with native media; general CLI |
| BM25 (Caption) | `rag_bm25_caption` | Built in | BM25 over complete sessions with frozen captions |
| BM25 (Native Media) | `rag_bm25_native` | Built in | BM25 over session text, then complete selected native media; no visual embeddings |
| RAG-Dense | `rag_dense` | Built in | FastEmbed dense session retrieval; install recorded RAG dependencies |
| RAG-Hybrid | `rag_hybrid` | Built in | Dense and BM25 candidates fused by RRF |
| Mem0 (Mem0-O) | `mem0_oss` | Built in | Newer 2.0.2 mainline SDK with additive extraction and entity/lexical/vector retrieval |
| Mem0 (Mem0-A) | `mem0_agentic` | Built in | Legacy non-graph 1.0.6 SDK with fact extraction and action-based memory consolidation |
| A-MEM | `amem` | Built in | Bundled A-MEM SDK in its separate environment |
| MIRIX | `mirix` | Built in | Bundled MIRIX SDK and running memory service; aligned video/ASR assets required |
| Multimodal RAG (external adapter) | `multimodal_rag` | External adapter required | No built-in visual-embedding retrieval or multimodal fusion implementation |
| Oracle Caption | `oracle_caption` | External adapter required | No built-in oracle adapter; also requires a private evidence manifest |
| Oracle Multimodal | `oracle_multimodal` | External adapter required | No built-in oracle adapter; also requires a private evidence manifest |
| Text-Only Control | `text_only_shortcut` | External adapter required | Protocol-only control profile |
| Question-Only Control | `question_only` | External adapter required | Protocol-only control profile |

Built-in means adapter code is supplied. Install dependencies, configure models
and supply matching media/caption assets before running it. It does not certify
provider compatibility, model accuracy or reproduction of a published experiment.
API and local answer deployments share the input interface; memory-framework
services retain their own setup requirements.

Use `scripts/run_framework.py` for `mem0_agentic`, `mem0_oss`, `amem`, `mirix`, `rag_dense`,
`rag_hybrid` and `rag_bm25_caption`. Full-context and native BM25 profiles use the
general `beyond-recall-eval infer` CLI with `--adapter-file adapters/builtin.py`.
`mem0_agentic` and `mem0_oss` identify different methods. The Agentic adapter loads
`frameworks/Mem0-Agentic/` (legacy non-graph 1.0.6); OSS loads `frameworks/Mem0/`
(newer 2.0.2 mainline with its recorded local changes). The launcher verifies
source hashes, and loading another variant into the same process is rejected.
Both use the same prepared inputs and answering interface, but SDK extraction,
consolidation and retrieval algorithms remain different. See
[MEM0_VARIANTS.md](MEM0_VARIANTS.md) for installation and validation details.

For external profiles, supply `--adapter-file /path/to/adapter.py` to the general
CLI. The file must define `create_system(info, config)` returning a system with
`runtime_info()`, `ingest(session)`, `answer(question)` and `close()`. Follow
`examples/minimal_adapter.py` and the input contract in `INPUT_PROTOCOL.md`.
The minimal example is an interface demonstration and does not implement any
listed memory method. The built-in adapter rejects external-only profiles before
model calls rather than substituting another algorithm.

Historical engineering runs completed 20/20 questions for the previously bundled
Mem0 1.0.6 adapter, A-MEM,
dense RAG and hybrid RAG. MIRIX completed 2/20; 17 provider-balance errors and one
complete-video timeout remain unresolved. These checks do not establish answer
accuracy. Configuration profiles are not a list of fully tested methods.

The previous Mem0 check was labeled OSS but used the legacy 1.0.6 SDK. Its result
must not be presented as validation of the newly bundled 2.0.2 mainline. Current
variant checks are described in `MEM0_VARIANTS.md` and do not claim model accuracy.
