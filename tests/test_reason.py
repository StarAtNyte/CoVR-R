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
