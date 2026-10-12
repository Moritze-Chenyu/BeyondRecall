# Documentation

Run shell commands from the repository root unless stated otherwise.

| Topic | Guide |
|---|---|
| Installation and evaluation | [Quick Start](START_HERE.md) |
| Models and runtime settings | [Configuration](CONFIGURATION.md) |
| Available method implementations | [Methods](METHODS.md) |
| Mem0 Agentic and OSS | [Mem0 Variants](MEM0_VARIANTS.md) |
| Complete inputs and timestamp filtering | [Input Protocol](INPUT_PROTOCOL.md) |
| PDF evidence-page selection | [PDF Evidence Policy](PDF_EVIDENCE_POLICY.md) |
| Paper citation and BibTeX | [Citation](CITATION.md) |
| Original code and data licensing scope | [Licensing](LICENSING.md) |
| Bundled SDK sources and licenses | [Third-Party Components](THIRD_PARTY.md) |

The [dataset](../data/benchmark/README.md) contains conversation histories,
questions, reference answers and media references. The API template is at
`config/.env.example`; copy it to a private `.env` in the repository root.
Verify published files from the repository root with:

```bash
sha256sum -c checksums/SHA256SUMS
```

`README.md`, `.gitignore`, `pyproject.toml`, `LICENSE` and `CITATION.cff` remain
at the root for the repository landing page, Git rules, package installation,
license visibility and GitHub citation support.
