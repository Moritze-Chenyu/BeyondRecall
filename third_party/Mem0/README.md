# Mem0 OSS mainline runtime snapshot

This snapshot contains the current local Mem0 mainline SDK (2.0.2) used by the
historical `--oss-mainline` evaluation path, with its existing local compatibility
changes preserved. It is distinct from the legacy non-graph 1.0.6 SDK bundled
under `../Mem0-Agentic/`. See `SDK_SOURCE.json` and `LOCAL_CHANGES.patch`.

Install in a dedicated environment with `pip install --no-deps -e .` after the
recorded dependencies. The evaluation launcher selects and verifies the exact
source snapshot, so do not install both editable SDKs into the same environment.
