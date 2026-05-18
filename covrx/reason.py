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
