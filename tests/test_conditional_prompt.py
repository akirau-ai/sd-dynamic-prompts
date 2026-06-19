import pytest


class StubGenerator:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, template, num_prompts=1, seeds=None):
        return [self.outputs.get(template, template)]


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
