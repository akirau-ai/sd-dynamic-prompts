import pytest
import random


class StubGenerator:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, template, num_prompts=1, seeds=None):
        return [self.outputs.get(template, template)]


class CountingRandom(random.Random):
    def __init__(self):
        self.seed_calls = []
        super().__init__()

    def seed(self, a=None, version=2):
        self.seed_calls.append(a)
        return super().seed(a, version)


class SharedContextGenerator:
    def __init__(self):
        self._context = type("Ctx", (), {})()
        self._context.rand = CountingRandom()

    def generate(self, template, num_prompts=1, seeds=None):
        raise AssertionError("shared sampler path should not call generate()")

    def _choices_for(self, template):
        return {
            "{a|b|c}": ["a", "b", "c"],
            "{1|2|3}": ["1", "2", "3"],
        }[template]

    def _sample_prompts(self, template, num_prompts):
        choices = self._choices_for(template)
        for _ in range(num_prompts):
            yield choices[self._context.rand.randrange(len(choices))]

    @property
    def sample_prompts(self):
        return self._sample_prompts


def test_conditional_prompt_samples_assignments_in_shared_context():
    from sd_dynamic_prompts.conditional_prompt import _sample_assignments

    generator = SharedContextGenerator()
    generator._context.sample_prompts = generator.sample_prompts
    initial_seed_calls = len(generator._context.rand.seed_calls)

    variables = _sample_assignments(
        "${x={a|b|c}} ${y={1|2|3}}",
        prompt_generator=generator,
        num_prompts=1,
        seeds=[123],
    )

    assert len(generator._context.rand.seed_calls) == initial_seed_calls + 1
    assert generator._context.rand.seed_calls[-1] == 123
    assert set(variables) == {"x", "y"}


def test_conditional_prompt_selects_matching_branch():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${action={run|standing|__aaa__}}
${if action==run}
prompt A
${elsif action==standing}
prompt B
${elsif action==__aaa__}
prompt C
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator({"{run|standing|__aaa__}": "standing"}),
    )

    assert "prompt B" in rendered
    assert "prompt A" not in rendered
    assert "prompt C" not in rendered
    assert "${action=standing}" in rendered


def test_conditional_prompt_leaves_plain_variable_references_for_later_processing():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${chara={anegasaki nene|kobayakawa rinko}}
${if chara==anegasaki nene}
${chara}, prompt A
${else}
${chara}, prompt B
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator(
            {"{anegasaki nene|kobayakawa rinko}": "anegasaki nene"},
        ),
    )

    assert "${chara=anegasaki nene}" in rendered
    assert "${chara}, prompt A" in rendered
    assert "prompt B" not in rendered


def test_conditional_prompt_uses_else_branch():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${action={run|standing}}
${if action==sleep}
prompt A
${elsif action==walk}
prompt B
${else}
prompt C
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator({"{run|standing}": "run"}),
    )

    assert "prompt C" in rendered


def test_conditional_prompt_rejects_invalid_else_order():
    from sd_dynamic_prompts.conditional_prompt import (
        ConditionalPromptError,
        preprocess_conditional_prompt,
    )

    template = """
${action=run}
${if action==run}
prompt A
${else}
prompt B
${elsif action==standing}
prompt C
${endif}
""".strip()

    with pytest.raises(ConditionalPromptError):
        preprocess_conditional_prompt(
            template,
            prompt_generator=StubGenerator({"run": "run"}),
        )


def test_conditional_prompt_supports_not_equals_and_and():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${action={run|standing}}
${clothes={bikini|uniform}}
${if action==run AND clothes!=uniform}
prompt A
${elsif action==standing AND clothes==uniform}
prompt B
${else}
prompt C
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator(
            {
                "{run|standing}": "run",
                "{bikini|uniform}": "bikini",
            },
        ),
    )

    assert "prompt A" in rendered
    assert "prompt B" not in rendered
    assert "prompt C" not in rendered


def test_conditional_prompt_allows_swapped_comparison_operands():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${action={oral|vaginal}}
${if oral==action}
prompt A
${elsif vaginal==action}
prompt B
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator({"{oral|vaginal}": "oral"}),
    )

    assert "prompt A" in rendered
    assert "prompt B" not in rendered


def test_conditional_prompt_supports_contains_operator():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${clothes={micro bikini|school uniform}}
${if clothes*=bikini}
prompt A
${else}
prompt B
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator({"{micro bikini|school uniform}": "micro bikini"}),
    )

    assert "prompt A" in rendered
    assert "prompt B" not in rendered


def test_conditional_prompt_supports_not_contains_operator():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${clothes={micro bikini|school uniform}}
${if clothes!*=uniform}
prompt A
${else}
prompt B
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator({"{micro bikini|school uniform}": "micro bikini"}),
    )

    assert "prompt A" in rendered
    assert "prompt B" not in rendered


def test_conditional_prompt_supports_contains_with_and():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${action={run|standing}}
${clothes={micro bikini|school uniform}}
${if action==run AND clothes*=bikini}
prompt A
${else}
prompt B
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator(
            {
                "{run|standing}": "run",
                "{micro bikini|school uniform}": "micro bikini",
            },
        ),
    )

    assert "prompt A" in rendered
    assert "prompt B" not in rendered


def test_conditional_prompt_treats_lowercase_and_as_plain_text():
    from sd_dynamic_prompts.conditional_prompt import preprocess_conditional_prompt

    template = """
${pose=hand and knee}
${if pose==hand and knee}
prompt A
${endif}
""".strip()

    rendered = preprocess_conditional_prompt(
        template,
        prompt_generator=StubGenerator({"hand and knee": "hand and knee"}),
    )

    assert "prompt A" in rendered


def test_conditional_prompt_rejects_unclosed_block():
    from sd_dynamic_prompts.conditional_prompt import ConditionalPromptError, preprocess_conditional_prompt

    template = """
${action=run}
${if action==run}
prompt A
""".strip()

    with pytest.raises(ConditionalPromptError):
        preprocess_conditional_prompt(
            template,
            prompt_generator=StubGenerator({"run": "run"}),
        )
