# Research Notes

## SOP:

add to `requests_1.json`
run `python data/build_requests.py 1`
`python data/scripts/yelp_precompute_groundtruth.py selection_1`

## Yelp Request Generation
> also check `data/yelp/notes_1`

**Bold** is done in `data/yelp/request_1.json` (easy read) and `data/yelp/request_1.jsonl` (compact, used in production)

- **G01 (R00--R04)**: Flat / Simple Metadata. Requests use only direct item metadata (e.g., attributes, categories) with flat AND composition.
- **G02 (R05--R09)**: Flat / Review Text. Requests rely exclusively on conditions grounded in review text, composed using flat ANY logic.
- **G03 (R10--R14)**: Flat / Computed Metadata. Requests use Simple Metadata + Computed item metadata requiring interpretation (e.g., hours, time-window constraints) with flat AND logic.
- G04 (R15--R19): Flat / Social Metadata. Requests use Simple Metadata + reviewer or review metadata (e.g., elite status, recency) with flat AND logic.

> ...>>>>>>><<<<<<<... just do above

- G05 (R20--R24): Flat / Simple Metadata + Review Text. Requests combine direct item metadata and review text conditions in flat AND/OR form.
- G06 (R25--R29): Flat / Simple Metadata + Review Text + Social Metadata. Requests integrate structured item metadata, unstructured review text, and social signals within a flat Boolean structure.
- G07 (R30--R34): Nested (Balanced) / Simple Metadata + Review Text. Requests use balanced nested Boolean expressions over simple item metadata and review text.
- G08 (R35--R39): Nested (Balanced) / Computed Metadata + Social Metadata. Requests use balanced nesting while combining computed item metadata and social signals.
- G09 (R40--R44): Nested (Skewed) / Simple Metadata + Review Text. Requests use skewed (left- or right-branching) nested structures over simple metadata and review text.
- G10 (R45--R49): Nested (Skewed) / Computed Metadata + Social Metadata. Requests combine skewed nesting with computed metadata and social signals.


"evidence": { "kind": "review_text" }