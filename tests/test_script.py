import pytest


@pytest.mark.parametrize("enable_hr", [True, False], ids=["yes_hr", "no_hr"])
@pytest.mark.parametrize("is_combinatorial", [True, False], ids=["yes_comb", "no_comb"])
def test_script(
    monkeypatch,
    monkeypatch_webui,
    processing,
    enable_hr,
    is_combinatorial,
):
    from scripts.dynamic_prompting import Script

    s = Script()
    if not is_combinatorial:
        processing.batch_size = 3
    processing.set_prompt_for_test("{red|green|blue} ball")
    processing.set_negative_prompt_for_test("ugly")
    processing.enable_hr = enable_hr
    s.process(
        p=processing,
        is_enabled=True,
        is_combinatorial=is_combinatorial,
        combinatorial_batches=1,
        is_magic_prompt=False,
        is_feeling_lucky=False,
        is_attention_grabber=False,
        min_attention=0,
        max_attention=1,
        magic_prompt_length=0,
        magic_temp_value=1,
        use_fixed_seed=False,
        unlink_seed_from_prompt=False,
        disable_negative_prompt=False,
        enable_jinja_templates=False,
        enable_conditional_prompt=False,
        no_image_generation=False,
        max_generations=0,
        magic_model="magic",
        magic_blocklist_regex=None,
    )
    assert isinstance(processing.all_prompts, list)
    assert isinstance(processing.all_negative_prompts, list)
    assert isinstance(processing.all_hr_prompts, list)
    assert isinstance(processing.all_hr_negative_prompts, list)

    if is_combinatorial:
        assert processing.all_prompts == ["red ball", "green ball", "blue ball"]
        assert processing.all_negative_prompts == ["ugly"] * 3

        if enable_hr:
            assert processing.all_hr_prompts == processing.all_prompts
            assert processing.all_hr_negative_prompts == processing.all_negative_prompts
    else:
        assert len(processing.all_prompts) == 3  # can't assert on the contents


@pytest.mark.parametrize(
    ("hr_prompt_raw", "mode_info", "expected"),
    [
        (
            False,
            ("Append", "{cinematic|dramatic}", None, None, None, None, None, False),
            ["red ball, cinematic", "green ball, dramatic", "blue ball, cinematic"],
        ),
        (
            True,
            ("Append", "{cinematic|dramatic}", None, None, None, None, None, False),
            ["{red|green|blue} ball, {cinematic|dramatic}"] * 3,
        ),
        (
            False,
            ("Default", "{cinematic|dramatic}", None, None, None, None, None, False),
            ["cinematic", "dramatic", "cinematic"],
        ),
        (
            False,
            ("Default", "", None, None, None, None, None, True),
            ["red ball", "green ball", "blue ball"],
        ),
    ],
    ids=["expanded_append", "raw_append", "expanded_default", "expanded_remove_fp_empty"],
)
def test_hr_prompt_mode(
    monkeypatch,
    monkeypatch_webui,
    processing,
    hr_prompt_raw,
    mode_info,
    expected,
):
    from scripts.dynamic_prompting import Script

    s = Script()
    processing.batch_size = 3
    processing.enable_hr = True
    processing.hr_prompt_raw = hr_prompt_raw
    processing.set_prompt_for_test("{red|green|blue} ball")
    processing.hr_prompt = mode_info[1]
    processing.all_hr_prompts = [processing.hr_prompt] * processing.batch_size
    processing.extra_generation_params["Hires prompt mode"] = mode_info
    processing.set_negative_prompt_for_test("ugly")

    s.process(
        p=processing,
        is_enabled=True,
        is_combinatorial=True,
        combinatorial_batches=1,
        is_magic_prompt=False,
        is_feeling_lucky=False,
        is_attention_grabber=False,
        min_attention=0,
        max_attention=1,
        magic_prompt_length=0,
        magic_temp_value=1,
        use_fixed_seed=False,
        unlink_seed_from_prompt=False,
        disable_negative_prompt=False,
        enable_jinja_templates=False,
        enable_conditional_prompt=False,
        no_image_generation=False,
        max_generations=0,
        magic_model="magic",
        magic_blocklist_regex=None,
    )

    assert processing.all_hr_prompts == expected
