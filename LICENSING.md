# Licensing scope and pending decision

**Status: owner decision pending.** The top-level [LICENSE](LICENSE) is a status
notice, not an approved open license. Do not describe this repository's original
code or benchmark data as openly licensed until the owner selects a license.

| Material | Scope | Current status |
|---|---|---|
| Original evaluator, adapters, scripts, configurations and documentation | Repository-authored contributions outside `third_party/` and `data/` | Owner has not selected an open license |
| Original benchmark questions, reference answers and annotations | Only contributions for which the owner can grant rights | Owner has not selected a data license |
| Conversation text adapted from upstream datasets | Includes retained or adapted source content | Follow the applicable upstream dataset terms; no blanket relicensing |
| Mem0, A-Mem and MIRIX SDK snapshots | `third_party/`, including retained notices | Existing upstream licenses remain in effect; see `THIRD_PARTY.md` |
| Referenced images, PDFs, videos and audio | Distributed separately, not in this repository | Source-specific rights and redistribution terms remain applicable |

## Proposed option for owner review

Use [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
for original code and documentation, and
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
for original benchmark contributions that the owner is authorized to license.
This proposal has **not** been adopted. An alternative code license is MIT.
The complete Apache license text is already available in the bundled upstream
SDK at `third_party/Mem0/LICENSE`; its current grant covers that SDK, not the
original Beyond Recall code.

Choosing an original-data license does not automatically establish permission to
redistribute upstream source material. Preserve source attribution and record
applicable source licenses when publishing the corresponding data release.
The current source-release metadata does not establish one uniform license
for all upstream conversation text and media.

Once the owner selects a scheme, replace the pending top-level notice with the
chosen code license, add a separate data license with this explicit scope, and
update package metadata and the README. Citation metadata is independent of
this decision; see [CITATION.md](CITATION.md).
