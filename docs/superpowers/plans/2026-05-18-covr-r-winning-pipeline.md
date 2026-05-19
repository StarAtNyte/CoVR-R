# CoVR-R Winning Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-shot Reason→Describe→Contrastive-Retrieve→Rerank pipeline that beats the CoVR-R baseline (Qwen3-VL hidden-state retrieval) on Recall@{1,5,10,50}, runnable on Modal A100-80GB.

**Architecture:** Reuse the authors' baseline harness (`data.py` record/pool loading, `evaluate.compute_recall`, Qwen3-VL reasoning) but replace the fragile LLM-hidden-state retrieval space with a contrastively-trained video-text dual encoder, and add one VLM pointwise rerank pass over top-50. A new `covrx/` package wraps the baseline; Modal runs the GPU stages.

**Tech Stack:** Python 3.11, PyTorch, transformers, qwen-vl-utils, open_clip_torch (default contrastive encoder), Modal, pytest, numpy.

---

## Reused baseline (read-only, imported via sys.path)

Located at `/home/nitiz/Projects/Others/CVPR/CoVR-R/CoVR-R-authors-github/` (flat modules: `config`, `data`, `model`, `prompts`, `evaluate`). We import and reuse:
- `data.Record`, `data.load_records`, `data.list_videos` — record + pool parsing (handles merged webvid/ss2, `.mp4`/`.webm` ext).
- `evaluate.compute_recall(scores_q2t, gt_indices)` — Recall@{1,5,10,50}.
- `model.VideoProcessor` + `prompts.get_prompts("reasoning")` — Qwen3-VL load + reasoning prompts (reasoning/rerank stages only).

We do NOT modify baseline files.

## Key constraints baked into the plan

- val/test JSON (`val-set_no-labels.json`, `test-set_no-labels.json`) has **no `video_target`** → no ground truth → recall cannot be computed locally on them. Leaderboard scores them. Local ablation uses a self-built dev set from **allowed external labeled data** (WebVid-CoVR public train triplets, which have targets). Rules forbid training on val/test only; external labeled data is allowed for dev/ablation, not training.
- Pool isolation: `webvid` query retrieves only among WebVid videos; `ss2` only among SS2. Driven by `Record.video_extension` (`.mp4`=webvid, `.webm`=ss2) per baseline `_load_merged_json_records`.
- Zero-shot: all models frozen. Generated descriptions disclosed in submission comment.

## File structure

```
covrx/
  __init__.py
  _baseline.py        # sys.path shim → import baseline data/evaluate/model/prompts
  config.py           # CovrxConfig dataclass
  pools.py            # split records by pool; no-labels-safe loading
  frames.py           # uniform frame sampling from a video file
  encoder.py          # ContrastiveEncoder ABC + registry + DummyEncoder + OpenClipEncoder
  reason.py           # ReasonedDescriber: Qwen3-VL structured R + self-consistency → target text
  index.py            # build per-pool gallery .npz {video_basename: vec}
  fusion.py           # score normalization + dual-sim / rerank fusion
  rerank.py           # VLM pointwise reranker over top-K
  retrieve.py         # pool-scoped cosine, recall (dev), submission JSON (val/test)
infra/
  modal_app.py        # Modal app: A100-80GB cls, Volume cache, .map() batch
configs/
  dev.yaml            # local dev-set ablation config
  val.yaml            # val-set_no-labels submission config
  test.yaml           # test-set_no-labels submission config
tests/
  test_config.py test_pools.py test_frames.py test_encoder.py
  test_index.py test_fusion.py test_rerank.py test_retrieve.py
docs/superpowers/specs/2026-05-18-covr-r-winning-approach-design.md   # existing spec
```

Testing strategy: pure-logic units (config, pools, frames, fusion, retrieve, index with `DummyEncoder`, rerank with fake scorer) get strict pytest TDD. GPU/Modal units (real `OpenClipEncoder`, `ReasonedDescriber`, `modal_app`) get a CPU-tiny smoke test plus a documented manual `modal run` verification — a 32B VLM cannot be unit-tested in-process; the plan says so explicitly rather than faking it.

---

### Task 0: Repo + package scaffold

**Files:**
- Create: `.gitignore`, `pytest.ini`, `requirements-covrx.txt`, `covrx/__init__.py`, `covrx/_baseline.py`
- Test: `tests/test_baseline_import.py`

- [ ] **Step 1: Init git repo at project root**

The top-level dir is not a git repo; the baseline subdir has its own nested `.git`. Init top-level and exclude the baseline checkout from tracking (we import it at runtime, don't version it).

Run:
```bash
cd /home/nitiz/Projects/Others/CVPR/CoVR-R
git init
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
CoVR-R-authors-github/
__pycache__/
*.pyc
.pytest_cache/
artifacts/
videos/
*.npz
.venv/
modal_volume_local/
```

- [ ] **Step 3: Write `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: Write `requirements-covrx.txt`**

```
torch>=2.4.0
transformers>=4.40.0
qwen-vl-utils>=0.0.14
open_clip_torch>=2.24.0
av>=16.0.0
decord>=0.6.0
numpy>=1.24.0
pyyaml>=6.0.0
modal>=0.64.0
pytest>=8.0.0
```

- [ ] **Step 5: Write `covrx/__init__.py`**

```python
"""CoVR-R winning pipeline package."""
```

- [ ] **Step 6: Write `covrx/_baseline.py`**

```python
"""Shim to import the authors' baseline modules without modifying them."""
import os
import sys

BASELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "CoVR-R-authors-github",
)
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

import data as baseline_data          # noqa: E402
import evaluate as baseline_evaluate  # noqa: E402

Record = baseline_data.Record
load_records = baseline_data.load_records
list_videos = baseline_data.list_videos
compute_recall = baseline_evaluate.compute_recall
```

- [ ] **Step 7: Write `tests/test_baseline_import.py`**

```python
from covrx._baseline import Record, load_records, compute_recall


def test_baseline_symbols_importable():
    assert callable(load_records)
    assert callable(compute_recall)
    rec = Record(index=0, reference_token="a", target_token="b",
                 edit_instruction="x", video_path="/tmp/a.mp4")
    assert rec.reference_key == "a.mp4"
    assert rec.target_key == "b.mp4"
```

- [ ] **Step 8: Run test, expect PASS**

Run: `python -m pytest tests/test_baseline_import.py -v`
Expected: PASS (baseline import works via sys.path shim).

- [ ] **Step 9: Commit**

```bash
git add .gitignore pytest.ini requirements-covrx.txt covrx tests
git commit -m "chore: scaffold covrx package + baseline import shim"
```

---

### Task 1: CovrxConfig

**Files:**
- Create: `covrx/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test `tests/test_config.py`**

```python
import textwrap
from covrx.config import CovrxConfig, load_covrx_config


def test_defaults():
    c = CovrxConfig()
    assert c.vlm_model == "Qwen/Qwen3-VL-32B-Instruct"
    assert c.encoder == "openclip"
    assert c.rerank_enabled is True
    assert c.dual_sim_enabled is False
    assert c.top_k == 50
    assert 0.0 <= c.fusion_alpha <= 1.0


def test_yaml_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        encoder: dummy
        rerank_enabled: false
        fusion_alpha: 0.7
        n_self_consistency: 3
    """))
    c = load_covrx_config(str(p))
    assert c.encoder == "dummy"
    assert c.rerank_enabled is False
    assert c.fusion_alpha == 0.7
    assert c.n_self_consistency == 3
    assert c.vlm_model == "Qwen/Qwen3-VL-32B-Instruct"  # untouched default
```

- [ ] **Step 2: Run test, expect FAIL** (`ModuleNotFoundError: covrx.config`)

Run: `python -m pytest tests/test_config.py -v`

- [ ] **Step 3: Write `covrx/config.py`**

```python
from dataclasses import dataclass, fields
from typing import Optional
import yaml


@dataclass
class CovrxConfig:
    # data
    label_path: str = ""
    video_dir: str = "./videos"
    has_labels: bool = False              # True only for self-built dev set

    # reasoning VLM
    vlm_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    vlm_fallback_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    sample_fps: float = 1.0
    reasoning_tokens: int = 160
    description_tokens: int = 220
    n_self_consistency: int = 3
    sc_temperature: float = 0.9
    synth_temperature: float = 0.5

    # contrastive encoder
    encoder: str = "openclip"             # registry key: openclip|dummy|...
    encoder_model: str = "ViT-SO400M-16-SigLIP2-384"
    encoder_pretrained: str = "webli"     # resolve exact tag in Task 3 step 1
    num_frames: int = 8

    # retrieval / fusion / rerank
    top_k: int = 50
    fusion_alpha: float = 0.6             # final = a*retrieval + (1-a)*rerank
    dual_sim_enabled: bool = False
    dual_sim_weight: float = 0.3
    rerank_enabled: bool = True

    # io
    artifact_dir: str = "./artifacts/covrx"
    gallery_npz: str = "./artifacts/covrx/gallery.npz"
    submission_path: str = "./artifacts/covrx/submission.json"


def load_covrx_config(yaml_path: str) -> CovrxConfig:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = CovrxConfig()
    valid = {fld.name for fld in fields(CovrxConfig)}
    for k, v in data.items():
        if k in valid:
            setattr(cfg, k, v)
    return cfg
```

- [ ] **Step 4: Run test, expect PASS**

Run: `python -m pytest tests/test_config.py -v`

- [ ] **Step 5: Commit**

```bash
git add covrx/config.py tests/test_config.py
git commit -m "feat: CovrxConfig + yaml loader"
```

---

### Task 2: Pool routing (no-labels-safe)

**Files:**
- Create: `covrx/pools.py`
- Test: `tests/test_pools.py`

Baseline `load_records` needs `video_target` for `target_token`; no-labels JSON lacks it (token becomes `""`). We add a loader that (a) splits records into `webvid`/`ss2` pools by extension, (b) tolerates missing targets, (c) flags whether labels exist.

- [ ] **Step 1: Write failing test `tests/test_pools.py`**

```python
import json
from covrx.pools import load_pool_records, PoolRecord


def _write(p, obj):
    p.write_text(json.dumps(obj))
    return str(p)


def test_no_labels_merged(tmp_path):
    path = _write(tmp_path / "val.json", [
        {"webvid": [{"id": 0, "video_source": "112/1016223889",
                     "modification_text": "make it night"}]},
        {"ss2": [{"id": 1, "video_source": "777/abc",
                  "modification_text": "add a dog"}]},
    ])
    pools = load_pool_records(path, "./videos")
    assert set(pools) == {"webvid", "ss2"}
    wv = pools["webvid"][0]
    assert isinstance(wv, PoolRecord)
    assert wv.reference_token == "1016223889"
    assert wv.reference_key == "1016223889.mp4"
    assert wv.edit_instruction == "make it night"
    assert wv.has_label is False
    assert wv.target_token == ""
    ss = pools["ss2"][0]
    assert ss.reference_key == "abc.webm"


def test_with_labels_devset(tmp_path):
    path = _write(tmp_path / "dev.json", [
        {"webvid": [{"id": 5, "video_source": "1/x", "video_target": "1/y",
                     "modification_text": "edit"}]},
    ])
    pools = load_pool_records(path, "./videos")
    r = pools["webvid"][0]
    assert r.has_label is True
    assert r.target_token == "y"
    assert r.target_key == "y.mp4"
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_pools.py -v`

- [ ] **Step 3: Write `covrx/pools.py`**

```python
import json
import os
from dataclasses import dataclass
from typing import Dict, List

POOL_EXT = {"webvid": ".mp4", "ss2": ".webm"}


@dataclass
class PoolRecord:
    index: int
    pool: str
    reference_token: str
    target_token: str
    edit_instruction: str
    video_path: str
    ext: str

    @property
    def reference_key(self) -> str:
        return f"{self.reference_token}{self.ext}"

    @property
    def target_key(self) -> str:
        return f"{self.target_token}{self.ext}"

    @property
    def has_label(self) -> bool:
        return bool(self.target_token)


def load_pool_records(label_path: str, video_dir: str) -> Dict[str, List[PoolRecord]]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    pools: Dict[str, List[PoolRecord]] = {}
    for section in data:
        for pool_key, entries in section.items():
            ext = POOL_EXT.get(pool_key, ".mp4")
            bucket = pools.setdefault(pool_key, [])
            for i, row in enumerate(entries):
                src = str(row.get("video_source", "")).strip()
                tgt = str(row.get("video_target", "")).strip()
                ref_token = os.path.basename(src).split(".")[0]
                tgt_token = os.path.basename(tgt).split(".")[0] if tgt else ""
                bucket.append(PoolRecord(
                    index=int(row.get("id", row.get("index", i))),
                    pool=pool_key,
                    reference_token=ref_token,
                    target_token=tgt_token,
                    edit_instruction=(row.get("modification_text") or "").strip(),
                    video_path=os.path.join(video_dir, pool_key, f"{ref_token}{ext}"),
                    ext=ext,
                ))
    return pools
```

- [ ] **Step 4: Run test, expect PASS**

Run: `python -m pytest tests/test_pools.py -v`

- [ ] **Step 5: Commit**

```bash
git add covrx/pools.py tests/test_pools.py
git commit -m "feat: pool routing with no-labels-safe loading"
```

---

### Task 3: Contrastive encoder interface + impls

**Files:**
- Create: `covrx/encoder.py`
- Test: `tests/test_encoder.py`

- [ ] **Step 1: Resolve exact open_clip SigLIP2 model/pretrained tag from current docs**

Do NOT trust memorized tags. Run the find-docs skill (or `npx ctx7@latest`) for `open_clip_torch` query: "list_pretrained model names SigLIP2 create_model_and_transforms tokenizer". Record the exact `(model_name, pretrained)` pair that loads SigLIP2 SO400M and update `CovrxConfig.encoder_model` / `encoder_pretrained` defaults if they differ. Write the resolved pair into a comment at the top of `covrx/encoder.py`.

- [ ] **Step 2: Write failing test `tests/test_encoder.py`**

```python
import numpy as np
from covrx.encoder import get_encoder, ContrastiveEncoder


def test_dummy_encoder_contract():
    enc = get_encoder("dummy", model="x", pretrained="y", num_frames=4)
    assert isinstance(enc, ContrastiveEncoder)
    v = enc.encode_video_frames([np.zeros((8, 8, 3), np.uint8)] * 4)
    t = enc.encode_text("a man rides a skateboard at night")
    assert v.shape == t.shape == (enc.dim,)
    # L2-normalized
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4
    assert abs(np.linalg.norm(t) - 1.0) < 1e-4


def test_dummy_text_is_deterministic():
    enc = get_encoder("dummy", model="x", pretrained="y", num_frames=4)
    a = enc.encode_text("same string")
    b = enc.encode_text("same string")
    assert np.allclose(a, b)
```

- [ ] **Step 3: Run test, expect FAIL**

Run: `python -m pytest tests/test_encoder.py -v`

- [ ] **Step 4: Write `covrx/encoder.py`**

```python
# Resolved open_clip tag (Task 3 step 1): <FILL FROM DOCS BEFORE EDITING>
import hashlib
from abc import ABC, abstractmethod
from typing import List
import numpy as np

_REGISTRY = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


class ContrastiveEncoder(ABC):
    dim: int = 0

    @abstractmethod
    def encode_video_frames(self, frames: List[np.ndarray]) -> np.ndarray: ...

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-8
    return (x / n).astype(np.float32)


@register("dummy")
class DummyEncoder(ContrastiveEncoder):
    """Deterministic hash-based encoder for fast CPU tests. No model."""
    dim = 32

    def __init__(self, **_):
        pass

    def _hash_vec(self, key: str) -> np.ndarray:
        h = hashlib.sha256(key.encode()).digest()
        arr = np.frombuffer((h * 2)[: self.dim], dtype=np.uint8).astype(np.float32)
        return _l2(arr - 127.5)

    def encode_video_frames(self, frames):
        return self._hash_vec(f"v{len(frames)}-{int(frames[0].sum())}")

    def encode_text(self, text):
        return self._hash_vec(f"t{text}")


@register("openclip")
class OpenClipEncoder(ContrastiveEncoder):
    """SigLIP2 frame-mean video embedding + text embedding (shared space)."""

    def __init__(self, model: str, pretrained: str, num_frames: int = 8, device: str = "cuda"):
        import open_clip
        import torch
        self.torch = torch
        self.num_frames = num_frames
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained
        )
        self.model = self.model.eval().to(device)
        self.tokenizer = open_clip.get_tokenizer(model)
        with torch.no_grad():
            d = self.model.encode_text(self.tokenizer(["x"]).to(device)).shape[-1]
        self.dim = int(d)

    def encode_video_frames(self, frames):
        from PIL import Image
        torch = self.torch
        imgs = [self.preprocess(Image.fromarray(f)).unsqueeze(0) for f in frames]
        batch = torch.cat(imgs).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            v = feats.mean(0)
        return _l2(v.float().cpu().numpy())

    def encode_text(self, text):
        torch = self.torch
        toks = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            t = self.model.encode_text(toks)[0]
        return _l2(t.float().cpu().numpy())


def get_encoder(name: str, **kwargs) -> ContrastiveEncoder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
```

- [ ] **Step 5: Run test, expect PASS** (only `dummy` path exercised; `openclip` needs GPU)

Run: `python -m pytest tests/test_encoder.py -v`

- [ ] **Step 6: Commit**

```bash
git add covrx/encoder.py tests/test_encoder.py
git commit -m "feat: contrastive encoder interface + dummy + openclip impls"
```

---

### Task 4: Frame sampling

**Files:**
- Create: `covrx/frames.py`
- Test: `tests/test_frames.py`

- [ ] **Step 1: Write failing test `tests/test_frames.py`**

```python
import numpy as np
from covrx.frames import uniform_indices, sample_frames_from_array


def test_uniform_indices_basic():
    assert uniform_indices(100, 5) == [0, 24, 49, 74, 99]
    assert uniform_indices(3, 5) == [0, 1, 2, 2, 2]   # clamp when short
    assert uniform_indices(0, 5) == [0, 0, 0, 0, 0]


def test_sample_from_array():
    vid = np.arange(10 * 4 * 4 * 3, dtype=np.uint8).reshape(10, 4, 4, 3)
    out = sample_frames_from_array(vid, 4)
    assert len(out) == 4
    assert out[0].shape == (4, 4, 3)
    assert np.array_equal(out[0], vid[0])
    assert np.array_equal(out[-1], vid[9])
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_frames.py -v`

- [ ] **Step 3: Write `covrx/frames.py`**

```python
from typing import List
import numpy as np


def uniform_indices(n_total: int, n_want: int) -> List[int]:
    if n_want <= 0:
        return []
    if n_total <= 0 or n_want == 1:
        return [0] * n_want
    if n_total < n_want:                      # short clip: list then pad w/ last
        idx = list(range(n_total))
        return idx + [n_total - 1] * (n_want - n_total)
    return [int(i * (n_total - 1) / (n_want - 1)) for i in range(n_want)]


def sample_frames_from_array(video: np.ndarray, n_want: int) -> List[np.ndarray]:
    idx = uniform_indices(len(video), n_want)
    return [video[i] for i in idx]


def sample_frames(video_path: str, n_want: int) -> List[np.ndarray]:
    """Decode a video file and return n_want uniformly-spaced RGB frames."""
    import decord
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    idx = uniform_indices(len(vr), n_want)
    return [vr[i].asnumpy() for i in idx]
```

- [ ] **Step 4: Run test, expect PASS** (file-decode path not unit-tested; covered by Task 11 integration)

Run: `python -m pytest tests/test_frames.py -v`

- [ ] **Step 5: Commit**

```bash
git add covrx/frames.py tests/test_frames.py
git commit -m "feat: uniform frame sampling"
```

---

### Task 5: ReasonedDescriber (Qwen3-VL structured reasoning → target text)

**Files:**
- Create: `covrx/reason.py`
- Test: `tests/test_reason.py`

The describer is split into pure orchestration (testable with a fake LLM callable) and a thin Qwen3-VL backend. Orchestration: build structured R-schema prompt, run N self-consistency samples, synthesize, build final target description. We inject the generate function so logic is unit-tested.

- [ ] **Step 1: Write failing test `tests/test_reason.py`**

```python
from covrx.reason import ReasonedDescriber, STRUCT_SCHEMA


def test_structured_prompt_contains_schema_and_edit():
    rd = ReasonedDescriber(generate=lambda video, prompt, **k: "stub", n_self_consistency=1)
    p = rd.reasoning_prompt("change bicycle to skateboard, make it night")
    for facet in STRUCT_SCHEMA:
        assert facet in p
    assert "skateboard" in p


def test_describe_self_consistency_synthesizes():
    calls = []

    def fake_gen(video, prompt, **kw):
        calls.append(prompt)
        if "Synthesize" in prompt:
            return "SYNTH: skateboarder at night"
        if "post-edit" in prompt or "Describe the edited" in prompt:
            return "A person rides a skateboard down a street at night."
        return f"trace-{len(calls)}"

    rd = ReasonedDescriber(generate=fake_gen, n_self_consistency=3)
    out = rd.describe(video="/tmp/v.mp4",
                      edit="change bicycle to skateboard, make it night")
    assert out.target_description.startswith("A person rides a skateboard")
    assert out.reasoning.startswith("SYNTH")
    assert len(out.traces) == 3


def test_describe_single_trace_skips_synthesis():
    def fake_gen(video, prompt, **kw):
        if "Describe the edited" in prompt or "post-edit" in prompt:
            return "final desc"
        return "only-trace"

    rd = ReasonedDescriber(generate=fake_gen, n_self_consistency=1)
    out = rd.describe(video="/tmp/v.mp4", edit="add rain")
    assert out.reasoning == "only-trace"
    assert out.target_description == "final desc"
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_reason.py -v`

- [ ] **Step 3: Write `covrx/reason.py`**

```python
from dataclasses import dataclass
from typing import Callable, List, Optional

STRUCT_SCHEMA = ["states", "actions", "scene", "camera", "tempo"]

GenerateFn = Callable[..., str]


@dataclass
class DescribeResult:
    target_description: str
    reasoning: str
    traces: List[str]


class ReasonedDescriber:
    def __init__(self, generate: GenerateFn, n_self_consistency: int = 3):
        self.generate = generate
        self.n = max(1, n_self_consistency)

    def reasoning_prompt(self, edit: str) -> str:
        facets = ", ".join(STRUCT_SCHEMA)
        return (
            "You are reasoning about the after-effects of a video edit. "
            f'Edit instruction: "{edit}"\n\n'
            "Think step by step about the implied consequences (not just literal "
            "keywords). For each of these facets, state what the EDITED video must "
            f"show: {facets}.\n"
            "Cover object state transitions, temporal phase order, and "
            "cinematographic changes. Be concrete."
        )

    def _synthesis_prompt(self, edit: str, traces: List[str]) -> str:
        joined = "\n\n".join(f"Perspective {i+1}: {t}" for i, t in enumerate(traces))
        return (
            f'Edit instruction: "{edit}"\n\n'
            f"Multiple reasoning perspectives:\n{joined}\n\n"
            "Synthesize these into one consistent, concrete account of the key "
            "visual changes (2-3 sentences):"
        )

    def _description_prompt(self, edit: str, reasoning: str) -> str:
        return (
            f"Reasoning about the edit: {reasoning}\n\n"
            f'Now write a single vivid present-tense paragraph describing the '
            f'post-edit video after applying: "{edit}". Describe the edited video '
            f"only; cover subjects, actions, scene, lighting, camera, pacing."
        )

    def describe(self, video: str, edit: str) -> DescribeResult:
        traces: List[str] = []
        rprompt = self.reasoning_prompt(edit)
        for _ in range(self.n):
            t = self.generate(video=video, prompt=rprompt, temperature=0.9).strip()
            if t:
                traces.append(t)
        if not traces:
            traces = [edit]  # fallback: literal edit

        if len(traces) == 1:
            reasoning = traces[0]
        else:
            reasoning = self.generate(
                video=video, prompt=self._synthesis_prompt(edit, traces),
                temperature=0.5,
            ).strip() or traces[0]

        desc = self.generate(
            video=video, prompt=self._description_prompt(edit, reasoning),
            temperature=0.7,
        ).strip()
        if not desc:
            desc = reasoning
        return DescribeResult(target_description=desc, reasoning=reasoning, traces=traces)


def make_qwen_generate(processor, sample_fps: float) -> GenerateFn:
    """Thin Qwen3-VL backend (uses baseline VideoProcessor + evaluate._generate_text)."""
    from covrx._baseline import baseline_evaluate  # type: ignore
    from data import VideoDataset, collate_fn
    from torch.utils.data import DataLoader

    def generate(video: str, prompt: str, temperature: float = 0.7, **kw) -> str:
        ds = VideoDataset([video], prompt, sample_fps)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor.processor))
        for inputs, _ in loader:
            return baseline_evaluate._generate_text(
                processor, inputs, max_tokens=220,
                temperature=temperature, top_p=0.9, top_k=50, do_sample=True,
            )
        return ""

    return generate
```

Note: `covrx/_baseline.py` must also expose `baseline_evaluate`; add `baseline_evaluate = baseline_evaluate` is already imported there — add `baseline_evaluate` to its names. Update `_baseline.py` accordingly in this task.

- [ ] **Step 4: Extend `covrx/_baseline.py`** — add line:

```python
baseline_evaluate = baseline_evaluate  # re-export module for reason.py
```

(Place after the existing imports; it just makes the name explicit for `from covrx._baseline import baseline_evaluate`.)

- [ ] **Step 5: Run test, expect PASS**

Run: `python -m pytest tests/test_reason.py -v`

- [ ] **Step 6: Commit**

```bash
git add covrx/reason.py covrx/_baseline.py tests/test_reason.py
git commit -m "feat: ReasonedDescriber structured R + self-consistency"
```

---

### Task 6: Gallery indexer

**Files:**
- Create: `covrx/index.py`
- Test: `tests/test_index.py`

Builds `{video_basename_with_ext: vec}` npz per pool, resumable (skip keys already present). Encoder injected so tested with `DummyEncoder` and a fake frame loader.

- [ ] **Step 1: Write failing test `tests/test_index.py`**

```python
import numpy as np
from covrx.index import build_gallery
from covrx.encoder import get_encoder


def test_build_and_resume(tmp_path):
    enc = get_encoder("dummy")
    frames = {"a.mp4": [np.zeros((4, 4, 3), np.uint8)],
              "b.mp4": [np.ones((4, 4, 3), np.uint8)]}

    def fake_loader(path, n):
        return frames[path.split("/")[-1]]

    npz = tmp_path / "g.npz"
    n1 = build_gallery(["x/a.mp4", "x/b.mp4"], enc, str(npz),
                        frame_loader=fake_loader, num_frames=1)
    assert n1 == 2
    store = dict(np.load(str(npz)))
    assert set(store) == {"a.mp4", "b.mp4"}
    assert abs(np.linalg.norm(store["a.mp4"]) - 1.0) < 1e-4

    # resume: only c.mp4 is new
    frames["c.mp4"] = [np.full((4, 4, 3), 7, np.uint8)]
    n2 = build_gallery(["x/a.mp4", "x/b.mp4", "x/c.mp4"], enc, str(npz),
                        frame_loader=fake_loader, num_frames=1)
    assert n2 == 1   # only c encoded
    store = dict(np.load(str(npz)))
    assert set(store) == {"a.mp4", "b.mp4", "c.mp4"}
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_index.py -v`

- [ ] **Step 3: Write `covrx/index.py`**

```python
import os
from typing import Callable, Dict, List
import numpy as np

FrameLoader = Callable[[str, int], list]


def _load_existing(npz_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(npz_path):
        return {}
    with np.load(npz_path) as z:
        return {k: np.asarray(z[k], np.float32) for k in z.files}


def build_gallery(video_paths: List[str], encoder, npz_path: str,
                  frame_loader: FrameLoader, num_frames: int = 8) -> int:
    os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
    store = _load_existing(npz_path)
    encoded = 0
    for vp in video_paths:
        key = os.path.basename(vp)
        if key in store:
            continue
        try:
            frames = frame_loader(vp, num_frames)
            store[key] = encoder.encode_video_frames(frames).astype(np.float32)
            encoded += 1
        except Exception as e:  # missing/corrupt video → skip + log
            print(f"[index] skip {vp}: {e}")
        if encoded and encoded % 200 == 0:
            np.savez_compressed(npz_path, **store)
    np.savez_compressed(npz_path, **store)
    return encoded
```

- [ ] **Step 4: Run test, expect PASS**

Run: `python -m pytest tests/test_index.py -v`

- [ ] **Step 5: Commit**

```bash
git add covrx/index.py tests/test_index.py
git commit -m "feat: resumable per-pool gallery indexer"
```

---

### Task 7: Retrieval + recall + submission

**Files:**
- Create: `covrx/retrieve.py`
- Test: `tests/test_retrieve.py`

Pool-scoped cosine. Mask the reference video (cannot be its own target). Dev mode (has labels) → `compute_recall`. Val/test mode → submission JSON of ranked target tokens.

- [ ] **Step 1: Write failing test `tests/test_retrieve.py`**

```python
import numpy as np
from covrx.retrieve import rank_pool, recall_from_ranks, build_submission
from covrx.pools import PoolRecord


def _rec(idx, ref, tgt, ext=".mp4"):
    return PoolRecord(idx, "webvid", ref, tgt, "edit",
                      f"./videos/webvid/{ref}{ext}", ext)


def test_rank_masks_reference_and_orders():
    gallery = {"a.mp4": np.array([1., 0.], np.float32),
               "b.mp4": np.array([0., 1.], np.float32),
               "c.mp4": np.array([0.7, 0.7], np.float32)}
    q = np.array([1., 0.], np.float32)
    rec = _rec(0, "a", "c")             # ref=a must be masked out
    ranked = rank_pool(q, rec, gallery)
    assert ranked[0] != "a.mp4"
    assert ranked[0] == "c.mp4"         # closest non-ref


def test_recall_from_ranks():
    ranked_lists = [["c.mp4", "b.mp4"], ["x.mp4", "y.mp4", "z.mp4"]]
    gt = ["c.mp4", "z.mp4"]
    m = recall_from_ranks(ranked_lists, gt)
    assert m["R1"] == 50.0
    assert m["R5"] == 100.0


def test_build_submission_shape():
    sub = build_submission({"webvid": [(_rec(0, "a", ""), ["b.mp4", "c.mp4"])]})
    assert sub["webvid"][0]["id"] == 0
    assert sub["webvid"][0]["ranking"] == ["b", "c"]
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_retrieve.py -v`

- [ ] **Step 3: Write `covrx/retrieve.py`**

```python
import os
from typing import Dict, List, Tuple
import numpy as np

from covrx.pools import PoolRecord


def rank_pool(query_vec: np.ndarray, rec: PoolRecord,
              gallery: Dict[str, np.ndarray]) -> List[str]:
    keys = sorted(gallery)
    mat = np.stack([gallery[k] for k in keys]).astype(np.float32)
    sims = mat @ query_vec.astype(np.float32)
    if rec.reference_key in gallery:
        sims[keys.index(rec.reference_key)] = -np.inf
    order = np.argsort(-sims)
    return [keys[i] for i in order]


def recall_from_ranks(ranked_lists: List[List[str]], gt_keys: List[str]) -> Dict[str, float]:
    ranks = []
    for ranked, gt in zip(ranked_lists, gt_keys):
        ranks.append(ranked.index(gt) if gt in ranked else 10**9)
    ranks = np.asarray(ranks)
    n = len(ranks)
    f = lambda k: round(100.0 * np.count_nonzero(ranks < k) / n, 4)
    return {"R1": f(1), "R5": f(5), "R10": f(10), "R50": f(50)}


def _token(key: str) -> str:
    return os.path.splitext(key)[0]


def build_submission(
    per_pool: Dict[str, List[Tuple[PoolRecord, List[str]]]],
    top_k: int = 50,
) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for pool, items in per_pool.items():
        out[pool] = [
            {"id": rec.index, "ranking": [_token(k) for k in ranked[:top_k]]}
            for rec, ranked in items
        ]
    return out
```

- [ ] **Step 4: Run test, expect PASS**

Run: `python -m pytest tests/test_retrieve.py -v`

- [ ] **Step 5: Verify against baseline recall (sanity)**

Add `tests/test_retrieve.py::test_matches_baseline_compute_recall`:

```python
def test_matches_baseline_compute_recall():
    from covrx._baseline import compute_recall
    # query 0 gt at rank 0, query 1 gt at rank 2
    scores = np.array([[0.9, 0.1, 0.2], [0.1, 0.2, 0.9]], np.float32)
    base = compute_recall(scores, [0, 2])
    ranked = [["g0", "g2", "g1"], ["g2", "g1", "g0"]]
    ours = recall_from_ranks(ranked, ["g0", "g2"])
    assert ours["R1"] == base["R1"]
```

Run: `python -m pytest tests/test_retrieve.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add covrx/retrieve.py tests/test_retrieve.py
git commit -m "feat: pool-scoped retrieval, recall, submission builder"
```

---

### Task 8: Score fusion (rerank + dual-sim flag)

**Files:**
- Create: `covrx/fusion.py`
- Test: `tests/test_fusion.py`

- [ ] **Step 1: Write failing test `tests/test_fusion.py`**

```python
import numpy as np
from covrx.fusion import minmax_norm, fuse_scores


def test_minmax_norm():
    out = minmax_norm(np.array([0., 5., 10.]))
    assert np.allclose(out, [0.0, 0.5, 1.0])
    flat = minmax_norm(np.array([3., 3., 3.]))
    assert np.allclose(flat, [0.0, 0.0, 0.0])


def test_fuse_retrieval_and_rerank():
    retr = np.array([0.0, 1.0])      # cand1 better by retrieval
    rr = np.array([1.0, 0.0])        # cand0 better by rerank
    fused = fuse_scores(retr, rr, alpha=0.5)
    assert np.isclose(fused[0], fused[1])
    fused_r = fuse_scores(retr, rr, alpha=1.0)
    assert fused_r[1] > fused_r[0]   # alpha=1 → pure retrieval


def test_dual_sim_combine():
    from covrx.fusion import combine_dual_sim
    primary = np.array([0.0, 1.0])
    secondary = np.array([1.0, 0.0])
    c = combine_dual_sim(primary, secondary, weight=0.0)
    assert np.allclose(c, primary)
    c2 = combine_dual_sim(primary, secondary, weight=1.0)
    assert np.allclose(c2, secondary)
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_fusion.py -v`

- [ ] **Step 3: Write `covrx/fusion.py`**

```python
import numpy as np


def minmax_norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fuse_scores(retrieval: np.ndarray, rerank: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * minmax_norm(retrieval) + (1.0 - alpha) * minmax_norm(rerank)


def combine_dual_sim(primary: np.ndarray, secondary: np.ndarray,
                     weight: float) -> np.ndarray:
    return (1.0 - weight) * primary + weight * secondary
```

- [ ] **Step 4: Run test, expect PASS**

Run: `python -m pytest tests/test_fusion.py -v`

- [ ] **Step 5: Commit**

```bash
git add covrx/fusion.py tests/test_fusion.py
git commit -m "feat: score fusion + dual-sim combine"
```

---

### Task 9: VLM pointwise reranker

**Files:**
- Create: `covrx/rerank.py`
- Test: `tests/test_rerank.py`

Reranks top-K by a VLM pointwise relevance score of (target description, candidate video). Scorer injected for tests. On scorer error → fall back to retrieval order (return zeros so fusion leaves order unchanged when alpha→1, and is robust when not).

- [ ] **Step 1: Write failing test `tests/test_rerank.py`**

```python
import numpy as np
from covrx.rerank import rerank_topk


def test_rerank_reorders_topk_only():
    ranked = ["a.mp4", "b.mp4", "c.mp4", "d.mp4"]
    retr_sims = {"a.mp4": 0.9, "b.mp4": 0.8, "c.mp4": 0.7, "d.mp4": 0.1}

    def scorer(desc, key):
        return {"a.mp4": 0.1, "b.mp4": 0.9, "c.mp4": 0.2}[key]

    out = rerank_topk("desc", ranked, retr_sims, scorer,
                       top_k=3, alpha=0.0)  # pure rerank within top-3
    assert out[:1] == ["b.mp4"]            # best reranked
    assert out[-1] == "d.mp4"             # tail untouched


def test_rerank_scorer_failure_falls_back():
    ranked = ["a.mp4", "b.mp4", "c.mp4"]
    retr = {"a.mp4": 0.9, "b.mp4": 0.5, "c.mp4": 0.1}

    def bad(desc, key):
        raise RuntimeError("vlm timeout")

    out = rerank_topk("d", ranked, retr, bad, top_k=3, alpha=0.6)
    assert out == ranked   # unchanged on failure
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `python -m pytest tests/test_rerank.py -v`

- [ ] **Step 3: Write `covrx/rerank.py`**

```python
from typing import Callable, Dict, List
import numpy as np

from covrx.fusion import fuse_scores

Scorer = Callable[[str, str], float]


def rerank_topk(description: str, ranked: List[str],
                retrieval_sims: Dict[str, float], scorer: Scorer,
                top_k: int, alpha: float) -> List[str]:
    head = ranked[:top_k]
    tail = ranked[top_k:]
    try:
        rr = np.array([float(scorer(description, k)) for k in head], np.float32)
    except Exception as e:
        print(f"[rerank] fallback (scorer failed): {e}")
        return ranked
    retr = np.array([retrieval_sims[k] for k in head], np.float32)
    fused = fuse_scores(retr, rr, alpha)
    order = np.argsort(-fused)
    return [head[i] for i in order] + tail


def make_qwen_scorer(processor, sample_fps: float, key_to_path: Dict[str, str]) -> Scorer:
    """Pointwise 0-1 relevance of a candidate video to the target description."""
    from covrx._baseline import baseline_evaluate
    from data import VideoDataset, collate_fn
    from torch.utils.data import DataLoader

    def scorer(description: str, key: str) -> float:
        prompt = (
            f'Target description: "{description}"\n\n'
            "Does this video match the target description? "
            "Answer ONLY a number from 0.0 (no match) to 1.0 (perfect match)."
        )
        ds = VideoDataset([key_to_path[key]], prompt, sample_fps)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor.processor))
        for inputs, _ in loader:
            txt = baseline_evaluate._generate_text(
                processor, inputs, max_tokens=8,
                temperature=0.0, top_p=1.0, top_k=1, do_sample=False)
            for tok in txt.replace(",", " ").split():
                try:
                    return max(0.0, min(1.0, float(tok)))
                except ValueError:
                    continue
        return 0.0

    return scorer
```

- [ ] **Step 4: Run test, expect PASS**

Run: `python -m pytest tests/test_rerank.py -v`

- [ ] **Step 5: Commit**

```bash
git add covrx/rerank.py tests/test_rerank.py
git commit -m "feat: VLM pointwise reranker with failure fallback"
```

---

### Task 10: Modal app

**Files:**
- Create: `infra/modal_app.py`, `infra/__init__.py`
- Test: manual `modal run` (documented; no unit test — needs Modal account + GPU)

- [ ] **Step 1: Confirm current Modal API from docs**

Run find-docs / `npx ctx7@latest docs /llmstxt/modal_llms_txt "@app.cls gpu A100-80GB modal.enter load model once Volume mount .map batch image uv_pip_install"` and reconcile any API drift before writing the file.

- [ ] **Step 2: Write `infra/__init__.py`** (empty file).

- [ ] **Step 3: Write `infra/modal_app.py`**

```python
import os
import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install_from_requirements(os.path.join(ROOT, "requirements-covrx.txt"))
    .add_local_dir(os.path.join(ROOT, "covrx"), "/root/covrx")
    .add_local_dir(os.path.join(ROOT, "CoVR-R-authors-github"), "/root/CoVR-R-authors-github")
)

app = modal.App("covrx")
cache = modal.Volume.from_name("covrx-cache", create_if_missing=True)
CACHE = "/cache"


@app.cls(gpu="A100-80GB", image=image, volumes={CACHE: cache},
         timeout=60 * 60 * 6, scaledown_window=300)
class Pipeline:
    cfg_yaml: str = modal.parameter()

    @modal.enter()
    def setup(self):
        os.environ["HF_HOME"] = f"{CACHE}/hf"
        import sys
        sys.path.insert(0, "/root")
        from covrx.config import load_covrx_config
        from covrx.encoder import get_encoder
        self.cfg = load_covrx_config(self.cfg_yaml)
        self.encoder = get_encoder(
            self.cfg.encoder, model=self.cfg.encoder_model,
            pretrained=self.cfg.encoder_pretrained,
            num_frames=self.cfg.num_frames)
        # Qwen3-VL loaded lazily in reasoning/rerank methods to bound memory.

    @modal.method()
    def index_pool(self, video_paths: list[str]) -> int:
        from covrx.index import build_gallery
        from covrx.frames import sample_frames
        return build_gallery(video_paths, self.encoder, self.cfg.gallery_npz,
                              frame_loader=sample_frames,
                              num_frames=self.cfg.num_frames)

    @modal.method()
    def reason_describe(self, items: list[dict]) -> list[dict]:
        """items: [{'video': path, 'edit': str, 'id': int, 'pool': str}]"""
        import sys
        sys.path.insert(0, "/root/CoVR-R-authors-github")
        from config import Config
        from model import VideoProcessor
        from covrx.reason import ReasonedDescriber, make_qwen_generate
        bcfg = Config(model_name=self.cfg.vlm_model, parallelism="mp")
        proc = VideoProcessor(bcfg)
        proc.load_model(0)
        rd = ReasonedDescriber(make_qwen_generate(proc, self.cfg.sample_fps),
                               self.cfg.n_self_consistency)
        out = []
        for it in items:
            r = rd.describe(it["video"], it["edit"])
            out.append({**it, "target_description": r.target_description,
                        "reasoning": r.reasoning})
        return out


@app.local_entrypoint()
def main(stage: str, cfg: str):
    p = Pipeline(cfg_yaml=cfg)
    if stage == "index":
        import sys
        sys.path.insert(0, os.path.join(ROOT, "covrx"))
        from covrx.config import load_covrx_config
        from covrx._baseline import list_videos
        c = load_covrx_config(cfg)
        print("indexed:", p.index_pool.remote(list_videos(c.video_dir)))
```

- [ ] **Step 4: Smoke verify (manual, documented)**

Run: `modal run infra/modal_app.py --stage index --cfg configs/dev.yaml`
Expected: container builds, A100-80GB acquired, prints `indexed: <N>`, `covrx-cache` Volume populated. Record output in the task notes. (No automated assertion — requires Modal account + GPU; this is the documented verification step.)

- [ ] **Step 5: Commit**

```bash
git add infra/modal_app.py infra/__init__.py
git commit -m "feat: Modal app (A100-80GB cls, Volume cache, index/reason methods)"
```

---

### Task 11: Local dev harness on allowed labeled data

**Files:**
- Create: `covrx/devset.py`, `scripts/run_dev_eval.py`, `configs/dev.yaml`
- Test: `tests/test_devset.py`

Rules forbid training on val/test but allow other data for dev/ablation. WebVid-CoVR public triplets (HF `webvid-covr`) have `video_target` → usable to measure recall locally end-to-end with the `dummy` encoder for a fast wiring test, and with the real encoder on GPU.

- [ ] **Step 1: Write failing test `tests/test_devset.py`**

```python
import json
from covrx.devset import to_pool_json


def test_to_pool_json_shape(tmp_path):
    rows = [{"id": 1, "video_source": "1/a", "video_target": "1/b",
             "modification_text": "edit one"}]
    out = to_pool_json(rows, pool="webvid")
    assert out == [{"webvid": rows}]
    p = tmp_path / "dev.json"
    p.write_text(json.dumps(out))
    from covrx.pools import load_pool_records
    rec = load_pool_records(str(p), "./videos")["webvid"][0]
    assert rec.has_label and rec.target_token == "b"
```

- [ ] **Step 2: Run test, expect FAIL**; Run: `python -m pytest tests/test_devset.py -v`

- [ ] **Step 3: Write `covrx/devset.py`**

```python
from typing import List, Dict


def to_pool_json(rows: List[dict], pool: str) -> list:
    """Wrap labeled external triplets into the merged pool JSON shape."""
    return [{pool: rows}]
```

- [ ] **Step 4: Write `configs/dev.yaml`**

```yaml
label_path: ./artifacts/covrx/devset.json
video_dir: ./videos_dev
has_labels: true
encoder: dummy            # fast wiring check; switch to openclip on GPU
vlm_model: Qwen/Qwen3-VL-8B-Instruct
n_self_consistency: 1
rerank_enabled: false
top_k: 50
gallery_npz: ./artifacts/covrx/dev_gallery.npz
artifact_dir: ./artifacts/covrx/dev
submission_path: ./artifacts/covrx/dev_submission.json
```

- [ ] **Step 5: Write `scripts/run_dev_eval.py`** (end-to-end glue, dummy-encoder CPU path)

```python
"""End-to-end dev eval: pools -> (fake/real) describe -> encode -> retrieve -> recall."""
import sys
import numpy as np
from covrx.config import load_covrx_config
from covrx.pools import load_pool_records
from covrx.encoder import get_encoder
from covrx.index import build_gallery
from covrx.retrieve import rank_pool, recall_from_ranks
from covrx._baseline import list_videos


def main(cfg_path: str):
    cfg = load_covrx_config(cfg_path)
    enc = get_encoder(cfg.encoder, model=cfg.encoder_model,
                      pretrained=cfg.encoder_pretrained, num_frames=cfg.num_frames)

    def loader(path, n):
        return [np.zeros((8, 8, 3), np.uint8)]  # dummy-encoder placeholder

    build_gallery(list_videos(cfg.video_dir), enc, cfg.gallery_npz,
                  frame_loader=loader, num_frames=1)
    gallery = dict(np.load(cfg.gallery_npz))
    pools = load_pool_records(cfg.label_path, cfg.video_dir)

    for pool, recs in pools.items():
        ranked_lists, gts = [], []
        for r in recs:
            q = enc.encode_text(r.edit_instruction)  # describe stub for dummy path
            ranked_lists.append(rank_pool(q, r, gallery))
            gts.append(r.target_key)
        print(pool, recall_from_ranks(ranked_lists, gts))


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 6: Run test + script smoke**

Run: `python -m pytest tests/test_devset.py -v` → PASS
Run (after creating a tiny 4-triplet `devset.json` + matching dummy videos dir): `python scripts/run_dev_eval.py configs/dev.yaml`
Expected: prints `webvid {'R1': ..., ...}` without error (wiring proven; numbers meaningless for dummy encoder).

- [ ] **Step 7: Commit**

```bash
git add covrx/devset.py scripts/run_dev_eval.py configs/dev.yaml tests/test_devset.py
git commit -m "feat: local dev harness on allowed labeled data"
```

---

### Task 12: Val/test submission runner + ablation grid

**Files:**
- Create: `scripts/run_submission.py`, `scripts/ablate.py`, `configs/val.yaml`, `configs/test.yaml`
- Test: `tests/test_submission_io.py`

- [ ] **Step 1: Write failing test `tests/test_submission_io.py`**

```python
import json
from scripts.run_submission import assemble_submission
from covrx.pools import PoolRecord


def test_assemble_submission_writes_expected_keys(tmp_path):
    rec = PoolRecord(7, "webvid", "ref", "", "edit",
                     "./v/webvid/ref.mp4", ".mp4")
    per_pool = {"webvid": [(rec, ["t1.mp4", "t2.mp4"])]}
    out = tmp_path / "sub.json"
    assemble_submission(per_pool, str(out), top_k=50)
    data = json.loads(out.read_text())
    assert data["webvid"][0]["id"] == 7
    assert data["webvid"][0]["ranking"][:2] == ["t1", "t2"]
```

- [ ] **Step 2: Run test, expect FAIL**; Run: `python -m pytest tests/test_submission_io.py -v`

- [ ] **Step 3: Write `scripts/run_submission.py`**

```python
"""Full GPU pipeline → submission JSON. Run via Modal for val/test."""
import json
import os
import sys
import numpy as np

from covrx.config import load_covrx_config
from covrx.pools import load_pool_records
from covrx.encoder import get_encoder
from covrx.frames import sample_frames
from covrx.index import build_gallery
from covrx.retrieve import rank_pool, build_submission
from covrx.rerank import rerank_topk
from covrx._baseline import list_videos


def assemble_submission(per_pool, out_path, top_k):
    sub = build_submission(per_pool, top_k=top_k)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sub, f)
    return sub


def run(cfg_path: str):
    cfg = load_covrx_config(cfg_path)
    enc = get_encoder(cfg.encoder, model=cfg.encoder_model,
                      pretrained=cfg.encoder_pretrained, num_frames=cfg.num_frames)

    # 1. gallery per pool (videos laid out videos/<pool>/<token>.<ext>)
    build_gallery(list_videos(cfg.video_dir), enc, cfg.gallery_npz,
                  frame_loader=sample_frames, num_frames=cfg.num_frames)
    gallery = dict(np.load(cfg.gallery_npz))
    key_to_path = {os.path.basename(p): p for p in list_videos(cfg.video_dir)}

    # 2. reasoning backend (Qwen3-VL via baseline VideoProcessor)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                    "CoVR-R-authors-github"))
    from config import Config
    from model import VideoProcessor
    from covrx.reason import ReasonedDescriber, make_qwen_generate
    from covrx.rerank import make_qwen_scorer
    bcfg = Config(model_name=cfg.vlm_model, parallelism="mp")
    proc = VideoProcessor(bcfg)
    proc.load_model(0)
    describer = ReasonedDescriber(make_qwen_generate(proc, cfg.sample_fps),
                                  cfg.n_self_consistency)
    scorer = (make_qwen_scorer(proc, cfg.sample_fps, key_to_path)
              if cfg.rerank_enabled else None)

    pools = load_pool_records(cfg.label_path, cfg.video_dir)
    per_pool = {}
    for pool, recs in pools.items():
        items = []
        for r in recs:
            desc = describer.describe(r.video_path, r.edit_instruction).target_description
            q = enc.encode_text(desc)
            ranked = rank_pool(q, r, gallery)
            if scorer is not None:
                keys = sorted(gallery)
                mat = np.stack([gallery[k] for k in keys])
                sims = {k: float(s) for k, s in zip(keys, mat @ q)}
                ranked = rerank_topk(desc, ranked, sims, scorer,
                                     cfg.top_k, cfg.fusion_alpha)
            items.append((r, ranked))
        per_pool[pool] = items
    assemble_submission(per_pool, cfg.submission_path, cfg.top_k)
    print("wrote", cfg.submission_path)


if __name__ == "__main__":
    run(sys.argv[1])
```

- [ ] **Step 4: Write `configs/val.yaml` and `configs/test.yaml`**

`configs/val.yaml`:
```yaml
label_path: ./val-set_no-labels.json
video_dir: ./videos
has_labels: false
encoder: openclip
vlm_model: Qwen/Qwen3-VL-32B-Instruct
vlm_fallback_model: Qwen/Qwen3-VL-8B-Instruct
n_self_consistency: 3
rerank_enabled: true
top_k: 50
fusion_alpha: 0.6
gallery_npz: /cache/gallery_val.npz
artifact_dir: /cache/covrx/val
submission_path: /cache/covrx/val_submission.json
```

`configs/test.yaml`: identical but `label_path: ./test-set_no-labels.json`, `gallery_npz: /cache/gallery_test.npz`, `submission_path: /cache/covrx/test_submission.json`.

- [ ] **Step 5: Write `scripts/ablate.py`**

```python
"""Grid over encoder, n_self_consistency, fusion_alpha, rerank on dev set.
Requires labeled dev set (Task 11). Prints a table; you pick the best config."""
import itertools
import sys
import numpy as np
from covrx.config import load_covrx_config
from covrx.pools import load_pool_records
from covrx.encoder import get_encoder
from covrx.retrieve import rank_pool, recall_from_ranks

GRID = {
    "encoder": ["openclip"],
    "n_self_consistency": [1, 3, 5],
    "fusion_alpha": [0.4, 0.6, 0.8],
    "rerank_enabled": [False, True],
}


def main(cfg_path: str, gallery_npz: str):
    base = load_covrx_config(cfg_path)
    gallery = dict(np.load(gallery_npz))
    pools = load_pool_records(base.label_path, base.video_dir)
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(keys, combo))
        # NOTE: real run plugs describer+rerank; here measure retrieval-only
        # ceiling per combo using cached query vecs (see scripts/run_submission).
        print(params, "-> run scripts/run_submission with these overrides, "
                       "then recall_from_ranks on dev labels")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

(Ablation is operationally a loop of `run_submission` on the labeled dev set with overrides, then `recall_from_ranks`; the script documents the grid and the procedure rather than re-implementing the GPU pipeline.)

- [ ] **Step 6: Run tests, expect PASS**

Run: `python -m pytest tests/test_submission_io.py -v`

- [ ] **Step 7: Commit**

```bash
git add scripts/run_submission.py scripts/ablate.py configs/val.yaml configs/test.yaml tests/test_submission_io.py
git commit -m "feat: submission runner + ablation grid"
```

---

### Task 13: Compliance + submission packaging

**Files:**
- Create: `SUBMISSION.md`
- Test: full test suite green + manual checklist

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 2: Write `SUBMISSION.md`**

```markdown
# CoVR-R Submission Notes

Method: zero-shot Reason→Describe→Contrastive-Retrieve→Rerank.
- Frozen Qwen3-VL (no fine-tuning) generates structured after-effect reasoning
  and a post-edit target description.
- Frozen contrastive video-text encoder (SigLIP2 frame-mean) embeds the
  target description (query) and all pool videos (gallery).
- Cosine retrieval within the sample's pool; reference video masked.
- Single Qwen3-VL pointwise rerank pass over top-50.

Generated data disclosure: model-generated target descriptions and reasoning
traces are used as intermediate text. No human labeling of test data. No
training/fine-tuning on val or test. External datasets used only for offline
dev/ablation (not for training).
```

- [ ] **Step 3: Manual compliance checklist (record in task notes)**

- [ ] No training on val/test (all models frozen).
- [ ] No human labeling of test data.
- [ ] Generated descriptions disclosed (SUBMISSION.md).
- [ ] Test submission set public.
- [ ] Datasets used only for this challenge.

- [ ] **Step 4: Commit**

```bash
git add SUBMISSION.md
git commit -m "docs: submission notes + compliance checklist"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** 4-stage pipeline → Tasks 5/6/7/9; Modal infra → Task 10; backbone A/B via registry → Tasks 3+12; dual-sim flag → Tasks 1/8 (config + `combine_dual_sim`); validation strategy + no-labels reality → Tasks 11/12; error handling (skip/fallback) → Tasks 6/9; compliance → Task 13. No uncovered spec section.
- **Placeholder scan:** one intentional `<FILL FROM DOCS>` in Task 3 step 1 — it is an explicit doc-resolution action step (avoids fabricating an open_clip pretrained tag), not a deferred TODO. Ablate script intentionally documents a procedure rather than duplicating the GPU pipeline (DRY).
- **Type consistency:** `PoolRecord` fields/properties consistent across Tasks 2/7/11/12; `ContrastiveEncoder.encode_video_frames/encode_text` consistent Tasks 3/6/12; `rerank_topk` / `fuse_scores` signatures consistent Tasks 8/9/12; `build_submission` shape consistent Tasks 7/12.

## Open operational note (not a plan gap)

Exact contrastive backbone is chosen empirically in Task 12 ablation (openclip default; InternVideo2.5/LanguageBind can be added as new `@register` impls without touching callers). 72B-AWQ vs 32B VLM decided by Modal memory at Task 10 smoke; `vlm_fallback_model` already in config.
