# CoVR-R 2026 — Winning Approach Design

Date: 2026-05-18
Status: Approved (design), pending user spec review
Challenge: Reasoned-Aware Composed Video Retrieval (CoVR-R), VidLLMs workshop 2026

## Goal

Top the CoVR-R test leaderboard on Recall@{1,5,10,50}. Zero-shot, rules-compliant
(no supervised/self-supervised training on val/test; other data + model-generated
text allowed if disclosed in submission comment).

## Baseline (organizers' own SOTA)

`CoVR-R-authors-github`: Reason-Then-Retrieve with frozen Qwen3-VL.
- Gallery + query both embedded as **Qwen3-VL generated-description penultimate
  hidden states + importance-weighted pooling**, cosine retrieval within dataset pool.
- Published: Qwen3-VL-8B R@1 49.88; 72B R@1 55.48. Pooling ablation swings
  R@1 49.9 → 1.5 (max/last) — the hidden-state embedding is the fragile link.
- Reasoning stage (after-effect R = {states, actions, scene, camera, tempo}) is
  the genuine contribution and is kept.

## Constraints (confirmed with user)

- Compute: moderate — single A100/H100-80GB or 2x40GB on Modal Labs.
- Timeline: weeks — full build + ablation grid on val phase.
- Codebase: fork + extend the authors' baseline (reuse data/eval/embedding harness).
- Strategy: strong single pipeline (not heavy ensemble); one rerank pass allowed.

## Approach (selected: A)

Attack the weak link: replace LLM-hidden-state retrieval space with a real
contrastively-trained video-text dual encoder. Keep & strengthen reasoning. Add
one cheap VLM rerank pass. Training-free CIR recipe (CIReVL/LDRE-style) adapted
to video; fully zero-shot.

### Pipeline

```
GALLERY (offline, per pool: webvid / ss2)
  video → frame sample → Contrastive Video-Text Encoder → L2-norm vec → cache .npz

QUERY (per triplet)
  ref video + edit
    → Qwen3-VL: structured after-effect reasoning  R={states,actions,scene,camera,tempo}
    → Qwen3-VL: synthesize post-edit target description (self-consistency: N traces → 1)
    → Text Encoder (same contrastive model) → query vec

RETRIEVE
  cosine(query vec, gallery vecs within that sample's pool) → top-50

RERANK (single pass)
  Qwen3-VL pointwise: score(target description, candidate frames) over top-50
  final = α·retrieval_sim + (1-α)·rerank_score → resort → R@{1,5,10,50}
```

Pool isolation enforced by JSON key: `webvid` queries retrieve only in the WebVid
embedding set; `ss2` only in SS2. No labels read. Generated descriptions disclosed
in submission comment per Fair Usage.

## Components

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `infra/modal_app.py` | Modal app: `gpu="A100-80GB"`, `@app.cls` + `@modal.enter` loads model once, persistent `modal.Volume` for HF weights + embedding cache, `.map()` batch over videos, memory snapshot for cold-start | modal, baseline modules |
| `encoder.py` | Pluggable contrastive backbone interface: `encode_video(frames)`, `encode_text(str)`. Swappable impls | torch, transformers |
| `reason.py` | Qwen3-VL after-effect reasoning + target-description synthesis; structured R schema; N-way self-consistency. Extends baseline `prompts.py` "reasoning" style | baseline model.py |
| `rerank.py` | Qwen3-VL pointwise scorer over top-K + score fusion (`final = α·sim + (1-α)·rerank`) | reason model |
| `index.py` | Gallery embed + per-pool cache, resumable. Fork of `generate_embeddings.py` | encoder, data.py |
| `retrieve_eval.py` | Pool-scoped cosine retrieval, Recall@{1,5,10,50}, submission JSON. Fork of `evaluate.py` | index outputs |
| `configs/*.yaml` | backbone id, frame count, N self-consistency, α, top-K, dual-sim flag | — |

Each unit independently testable: encoder via embed-then-cosine sanity check;
reason via fixed (video, edit) → inspect trace/description; rerank via synthetic
ranked list; eval via tiny labeled val subset → known Recall.

## Backbone selection (decided on val, not pre-committed)

Candidates fitting single 80GB: InternVideo2.5, LanguageBind-Video,
SigLIP2-so400m (frame mean + temporal pool). `encoder.py` interface lets us A/B
on `val-set_no-labels.json` and pick best R@1. Qwen3-VL: 32B-class default;
72B-AWQ if it fits 80GB with video tokens, else fall back to 32B.

## Validation strategy

Val phase open. Loop: build → measure R@{1,5,10,50} on val leaderboard →
ablation grid (backbone × N self-consistency × α × frame count × rerank on/off ×
dual-sim flag) → freeze best config → single strong test submission (test phase
has limited submissions; spend them only on the frozen best).

## Optional dual-sim fusion (flagged, off by default)

Fuse contrastive sim with baseline hidden-state sim via score normalization +
weighted sum or RRF. Built but gated; enabled in final only if val shows gain.
Stays a single pipeline (one config), not an ensemble of submissions.

## Error handling

- Missing/corrupt video → skip + log (baseline pattern), continue.
- Reasoning failure / empty trace → fall back to literal "apply edit" description.
- Rerank timeout/error → fall back to pure retrieval ranking.
- Resumable: skip already-embedded videos via cache presence check.

## Risks & mitigations

- Contrastive space may underperform on reasoning-heavy edits → dual-sim fallback flag.
- 72B-AWQ may not fit 80GB w/ video tokens → 32B fallback path in config.
- Modal cold-start cost → persistent weight Volume + memory snapshot + load-once `@modal.enter`.
- Backbone availability/licensing as of May 2026 → interface abstracts it; ≥2 candidates.

## Out of scope (YAGNI)

- Multi-submission ensembles / RRF across many encoders (user chose single pipeline).
- Any fine-tuning, LoRA, or training on challenge data (rules + zero-shot).
- Serving/UI — offline batch + leaderboard JSON only.

## Compliance checklist

- [ ] No training on val/test (zero-shot, frozen models).
- [ ] No human labeling of test data.
- [ ] Generated descriptions disclosed in submission comment.
- [ ] Test submission public (prize eligibility).
- [ ] Datasets used only for this challenge.
