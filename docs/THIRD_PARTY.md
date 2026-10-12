# Bundled evaluation frameworks

These snapshots contain the runtime code used by the evaluation adapters. They
are separate upstream projects, retain their original licenses and notices, and
are not presented as newly implemented memory algorithms. Dashboard/front-end
code, model weights, repository history and private runtime state are excluded.

| Component | Source snapshot | License file | Local changes |
|---|---|---|---|
| Mem0 Agentic (Mem0-A) | Legacy non-graph mem0ai 1.0.6 installed SDK | [LICENSE](../third_party/Mem0-Agentic/LICENSE) | JSON-response reminder in OpenAI SDK wrapper |
| Mem0 OSS (Mem0-O) | Newer mainline 2.0.2 at 8b11e0787ad3d22efd4a85666d019eaf3e339116 | [LICENSE](../third_party/Mem0/LICENSE) | Five existing local compatibility changes recorded in its patch |
| A-Mem | ceffb860f0712bbae97b184d440df62bc910ca8d | [LICENSE](../third_party/A-mem/LICENSE) | None in SDK source; adapter fixes live in `frameworks.py` |
| MIRIX | 51f3342d5366b0e215439581f92e0323227146af | [LICENSE](../third_party/MIRIX/LICENSE) | Six existing compatibility changes, preserved from the tested checkout |

Each directory has `SDK_SOURCE.json` with source hashes and upstream provenance.
Changed sources also have `LOCAL_CHANGES.patch`. The legacy Agentic snapshot records the original wheel SHA256
for comparison; this text-only distribution includes the patched source snapshot,
not the binary wheel. MIRIX includes its
full Python server, agents, tools, prompts and memory managers. The service wrapper
and additional compatibility work are in `scripts/serve_mirix.py`.

RAG uses our adapters in `src/beyond_recall_eval/frameworks.py`, with FastEmbed,
BM25 and RRF; FastEmbed is installed through the recorded dependencies. Mem0 Agentic
and Mem0 OSS are distinct methods, with separate bundled SDKs and adapters.
Agentic uses action-based consolidation; OSS uses the newer additive and hybrid
retrieval pipeline. See `MEM0_VARIANTS.md`.

Install MIRIX from its source directory in editable mode: its upstream sdist
manifest excludes server modules. Do not replace this snapshot with a client-only
wheel when running the local memory service.
