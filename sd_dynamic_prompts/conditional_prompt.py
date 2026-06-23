from __future__ import annotations

import re
from dataclasses import dataclass

from dynamicprompts.generators.promptgenerator import PromptGenerator


class ConditionalPromptError(ValueError):
    pass


@dataclass(frozen=True)
class Branch:
    kind: str
    condition: str | None
    body: str


@dataclass(frozen=True)
class AssignmentMatch:
    start: int
    end: int
    name: str
    expression: str


_BLOCK_PATTERN = re.compile(
    r"\$\{if\s+(?P<if_cond>[^}]+)\}(?P<body>.*?)\$\{endif\}",
    re.DOTALL,
)
_ELSIF_PATTERN = re.compile(r"\$\{elsif\s+([^}]+)\}")
_ELSE_PATTERN = re.compile(r"\$\{else\}")
_COMPARISON_PATTERN = re.compile(
    r"^\s*([A-Za-z_-][A-Za-z0-9_-]*)\s*(==|!=|\*=|!\*=)\s*(.*?)\s*$",
    re.DOTALL,
)
_ASSIGNMENT_START_PATTERN = re.compile(
    r"\$\{\s*([A-Za-z_-][A-Za-z0-9_-]*)\s*=",
    re.DOTALL,
)


def preprocess_conditional_prompt(
    template: str,
    *,
    prompt_generator: PromptGenerator,
    num_prompts: int = 1,
    seeds: list[int] | None = None,
) -> str:
    if not template or "${if " not in template:
        return template

    validate_conditional_prompt(template)
    variables = _sample_assignments(
        template,
        prompt_generator=prompt_generator,
        num_prompts=num_prompts,
        seeds=seeds,
    )
    materialized_template = _replace_assignments(template, variables)

    return _BLOCK_PATTERN.sub(
        lambda match: _resolve_block(match, variables=variables),
        materialized_template,
    )


def validate_conditional_prompt(template: str) -> None:
    if not template or "${if " not in template:
        return

    position = 0
    while True:
        match = _BLOCK_PATTERN.search(template, position)
        if match is None:
            break

        _parse_branches(
            if_condition=match.group("if_cond"),
            body=match.group("body"),
        )
        _validate_condition(match.group("if_cond"))
        for elsif_condition in _ELSIF_PATTERN.findall(match.group("body")):
            _validate_condition(elsif_condition)

        position = match.end()

    if "${if " in _BLOCK_PATTERN.sub("", template):
        raise ConditionalPromptError(
            "Conditional_Prompt block is not closed. `${if ...}` requires a matching `${endif}`.",
        )


def _resolve_block(
    match: re.Match[str],
    *,
    variables: dict[str, str],
) -> str:
    branches = _parse_branches(
        if_condition=match.group("if_cond"),
        body=match.group("body"),
    )

    for branch in branches:
        if branch.kind == "else":
            return branch.body
        if _evaluate_condition(branch.condition or "", variables):
            return branch.body

    return ""


def _parse_branches(*, if_condition: str, body: str) -> list[Branch]:
    branches: list[Branch] = []
    position = 0
    current_kind = "if"
    current_condition = if_condition

    markers: list[tuple[int, int, str, str | None]] = []
    for elsif_match in _ELSIF_PATTERN.finditer(body):
        markers.append((elsif_match.start(), elsif_match.end(), "elsif", elsif_match.group(1)))
    for else_match in _ELSE_PATTERN.finditer(body):
        markers.append((else_match.start(), else_match.end(), "else", None))
    markers.sort(key=lambda item: item[0])

    seen_else = False
    for start, end, kind, condition in markers:
        if seen_else:
            raise ConditionalPromptError("`${else}` must be the final branch in Conditional_Prompt.")

        branches.append(
            Branch(
                kind=current_kind,
                condition=current_condition,
                body=body[position:start],
            ),
        )
        position = end
        current_kind = kind
        current_condition = condition
        if kind == "else":
            seen_else = True

    branches.append(
        Branch(
            kind=current_kind,
            condition=current_condition,
            body=body[position:],
        ),
    )
    return branches


def _sample_assignments(
    template: str,
    *,
    prompt_generator: PromptGenerator,
    num_prompts: int,
    seeds: list[int] | None,
) -> dict[str, str]:
    variables: dict[str, str] = {}
    assignments = list(_iter_assignments(template))
    shared_sampler = _get_shared_assignment_sampler(
        prompt_generator=prompt_generator,
        seeds=seeds,
    )

    if shared_sampler is not None:
        for assignment in assignments:
            variables[assignment.name] = shared_sampler(assignment.expression)
        return variables

    for assignment in assignments:
        sampled = (
            prompt_generator.generate(
                assignment.expression,
                num_prompts,
                seeds=seeds,
            )
            or [""]
        )
        variables[assignment.name] = sampled[0]
    return variables


def _get_shared_assignment_sampler(
    *,
    prompt_generator: PromptGenerator,
    seeds: list[int] | None,
):
    context = getattr(prompt_generator, "_context", None)
    if context is None or not hasattr(context, "sample_prompts") or not hasattr(context, "rand"):
        return None

    if seeds:
        context.rand.seed(seeds[0])

    def sample(expression: str) -> str:
        prompts = context.sample_prompts(expression, 1)
        return str(next(iter(prompts), ""))

    return sample


def _iter_assignments(template: str):
    position = 0
    while True:
        start = template.find("${", position)
        if start == -1:
            return

        name_match = _ASSIGNMENT_START_PATTERN.match(template[start:])
        if name_match is None:
            position = start + 2
            continue

        name = name_match.group(1)
        expr_start = start + name_match.end()
        expr_end = _find_assignment_end(template, expr_start)
        if expr_end == -1:
            raise ConditionalPromptError(
                f"Unterminated Conditional_Prompt assignment for `{name}`.",
            )

        yield AssignmentMatch(
            start=start,
            end=expr_end + 1,
            name=name,
            expression=template[expr_start:expr_end].strip(),
        )
        position = expr_end + 1


def _find_assignment_end(template: str, expr_start: int) -> int:
    brace_depth = 0
    for index in range(expr_start, len(template)):
        char = template[index]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            if brace_depth == 0:
                return index
            brace_depth -= 1
    return -1


def _replace_assignments(template: str, variables: dict[str, str]) -> str:
    output: list[str] = []
    position = 0
    for assignment in _iter_assignments(template):
        output.append(template[position:assignment.start])
        output.append(f"${{{assignment.name}={variables[assignment.name]}}}")
        position = assignment.end

    output.append(template[position:])
    return "".join(output)


def _evaluate_condition(condition: str, variables: dict[str, str]) -> bool:
    parts = _split_and_conditions(condition)
    if not parts or any(not part for part in parts):
        raise ConditionalPromptError(
            f"Unsupported Conditional_Prompt expression: {condition!r}.",
        )

    return all(_evaluate_comparison(part, variables) for part in parts)


def _evaluate_comparison(condition: str, variables: dict[str, str]) -> bool:
    comparison = _COMPARISON_PATTERN.match(condition)
    if comparison is None:
        raise ConditionalPromptError(
            f"Unsupported Conditional_Prompt expression: {condition!r}. "
            "Supported forms are `name==value`, `name!=value`, `name*=value`, `name!*=value`, and uppercase `AND` combinations.",
        )

    left_value, operator, right_value = comparison.groups()
    left = left_value.strip()
    right = right_value.strip()

    if left in variables:
        actual = variables[left].strip()
        expected = right
    elif right in variables:
        actual = left
        expected = variables[right].strip()
    else:
        raise ConditionalPromptError(
            "Conditional_Prompt comparison must reference a defined variable. "
            f"Got {condition!r}. Defined variables: {', '.join(sorted(variables)) or '(none)'}.",
        )

    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == "*=":
        return expected in actual
    if operator == "!*=":
        return expected not in actual
    raise ConditionalPromptError(f"Unsupported Conditional_Prompt operator: {operator!r}.")


def _validate_condition(condition: str) -> None:
    parts = _split_and_conditions(condition)
    if not parts or any(not part for part in parts):
        raise ConditionalPromptError(
            f"Unsupported Conditional_Prompt expression: {condition!r}.",
        )
    for part in parts:
        if _COMPARISON_PATTERN.match(part) is None:
            raise ConditionalPromptError(
                f"Unsupported Conditional_Prompt expression: {condition!r}. "
                "Supported forms are `name==value`, `name!=value`, `name*=value`, `name!*=value`, and uppercase `AND` combinations.",
            )


def _split_and_conditions(condition: str) -> list[str]:
    return [part.strip() for part in condition.split(" AND ")]
