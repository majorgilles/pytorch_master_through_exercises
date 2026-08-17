# Agent Instructions

## Repository purpose

This repository exists to build PyTorch mastery through progressive coding and mental exercises. When creating or changing a notebook, optimize for active prediction, deliberate practice, immediate private feedback, and repeatability—not passive exposition or answer memorization.

## Working and virgin notebook contract

Every subject must have a synchronized pair:

- `notebooks/<subject>_exercises.ipynb` — working notebook; may contain learner answers, outputs, and scratch work.
- `notebooks/<subject>_exercises_virgin.ipynb` — pristine retry notebook; answer cells contain comments only and all code-cell execution counts and outputs are cleared.

Never erase or replace a working notebook’s learner answers unless the user explicitly requests a reset. Never copy working answers into a virgin notebook.

Shared prompts, fixtures, tests, setup cells, and section documentation must remain synchronized between a pair. The only expected differences are learner answers, execution state, explicitly marked working/virgin title notes, and deliberate scratch cells excluded from the reusable curriculum.

After finalizing a new working notebook, remove accidental scratch cells and generate its pair with:

```bash
python scripts/create_virgin.py notebooks/<subject>_exercises.ipynb
```

The generator relies on answer-cell IDs matching `exercise-NNN-answer`.

## Creating a notebook for a new subject

### 1. Define the mastery target

Write one precise sentence describing what repeated coding and mental practice should make automatic. Build a progression from concrete one-operation cases to realistic machine-learning patterns and integrated capstones.

A strong progression usually includes:

1. vocabulary and rank-zero/rank-one foundations;
2. one concept at a time with small visible tensors;
3. shape or value prediction before execution;
4. natural misconceptions and deliberately invalid cases;
5. higher-rank combinations;
6. storage or autograd semantics where relevant;
7. realistic PyTorch patterns;
8. integrated capstones.

Use enough exercises to create repetition without padding the curriculum with duplicate cases. Give each section a clear conceptual purpose and explicit exercise range.

### 2. Create the working notebook

Name it:

```text
notebooks/<subject>_exercises.ipynb
```

Include:

- a title and concise usage instructions;
- a subject-specific mental checklist;
- a course map with section and exercise ranges;
- setup and private test helpers;
- progressive exercise groups;
- a completion standard;
- official PyTorch references.

### 3. Follow the four-cell exercise contract

Give each exercise stable, zero-padded IDs and place these cells in order:

1. `exercise-NNN-prompt` — Markdown with purpose, visible inputs, task, ingredients, required predictions/outputs, and next concept.
2. `exercise-NNN-fixture` — supplied input construction and private-reference registration.
3. `exercise-NNN-answer` — synchronized comments followed by blank learner space.
4. `exercise-NNN-test` — private-reference checks for every required variable.

Use stable IDs for title, checklist, map, setup/helper, section, completion, and reference cells as well.

## Learning-notebook documentation

For every exercise:

- Explain the purpose, inputs, expected learner outputs, and next concept in nearby Markdown.
- Add concise comments to supplied code and keep them synchronized with executable behavior.
- State what each axis represents when shape semantics matter.
- Explicitly distinguish elementwise operations, broadcasting, reductions, indexing, and matrix contractions.
- For probabilistic or loss code, distinguish training examples, candidate classes/tokens, and expected targets. State which indexed probability contributes directly to loss.
- State whether evaluation covers a complete dataset or a deliberately supplied test sequence.
- Keep explanations local to the exercise rather than front-loading all detail into one oversized introduction.
- Do not document future operations that are not yet present in the exercise.

For central mathematical relationships:

- Use standalone display LaTeX blocks.
- Define every symbol in surrounding prose.
- Avoid relying on inline LaTeX inside tables.
- Prefer a progression from a single element/example to the complete tensor/dataset relationship.
- Visually verify renderer-sensitive formulas such as piecewise definitions.

Documentation-only requests must not change executable behavior.

## Shape-prediction requirements

When shapes are part of the learning objective:

- Require literal Python tuples before value calculations.
- Name each prediction after the exact tensor, such as `ex027_y_shape`, not an ambiguous `out_shape` when the tensor is named `y`.
- Include relevant supplied inputs, prepared views, named intermediates, outputs, upstream gradients, and computed gradients.
- Explain aligned shapes as conceptual left-padding with ones; do not imply that alignment physically reshapes a tensor.
- If asking for expansion axes, define the axis numbering and expected tuple representation.

Do not reveal shape solutions in visible tests, fixture output, saved output, or assertion messages.

## Fixtures and private references

Visible fixtures must expose learner inputs and the forward operation needed for reasoning. They must not print expected shapes or values.

Expected outputs may be generated by generic helpers or a private support module named like:

```text
notebooks/_<subject>_fixtures.py
```

Keep required support modules beside their notebooks and make path lookup robust when Jupyter starts from the repository root or the `notebooks/` directory.

For manual-backpropagation curricula, private helpers may use autograd to construct references, but learner answer cells must not use `.backward()`, `torch.autograd.grad`, or `.grad` unless an exercise explicitly teaches autograd.

## Tests must not leak solutions

Tests should check presence, Python/tensor type, shape, dtype, and values while keeping expected answers private.

Do not write visible assertions containing answers, such as:

```python
_check_shape("ex038_dx_shape", (2, 3))
```

Instead, refer to private keys and fields:

```python
_check_private_value("ex038_dx_shape", "ex038", "dx_shape")
```

Failure messages should be actionable but non-revealing. Prefer “revisit right alignment” over displaying an expected tuple or tensor. Catch assertion output from `torch.testing.assert_close` when it would expose private expected values.

Do not save outputs from failed or passing tests in virgin notebooks.

## Answer-cell policy

New curricula must ship with blank learner answer cells. Comments may state required variable names and constraints, but must not contain executable solutions.

If maintaining a working notebook with existing learner progress:

- preserve the learner’s expressions and scratch work;
- fill only answers explicitly requested by the user;
- do not propagate those answers to the virgin pair;
- keep supplied scaffolding synchronized around the answers.

## Validation checklist

After creating or changing an exercise notebook:

1. Parse notebook JSON.
2. Confirm notebook format and required cell fields.
3. Confirm every cell ID is unique.
4. Compile every code cell.
5. Run setup and all visible fixtures sequentially in a fresh namespace.
6. Confirm private references provide every field requested by tests.
7. Simulate correct private answers and execute every supplied test.
8. Confirm visible test source contains no literal expected tuples or tensors.
9. Confirm fixtures do not print shape/value answers.
10. Confirm all answer cells in each virgin notebook have an empty Python AST body.
11. Confirm every virgin code cell has `execution_count = null` and `outputs = []`.
12. Compare working/virgin prompts, fixtures, and tests for synchronization.
13. Compile and lint private support modules and `scripts/create_virgin.py`.
14. Run `git diff --check`.

Use structured JSON scripts for broad notebook changes rather than fragile textual replacements. Validate immediately after every notebook rewrite.

## Adding the subject to the repository

After both notebooks and any support module pass validation:

- Add the working/virgin pair to the notebook table in `README.md`.
- Keep human-facing README content focused on goals, setup, practice, and retry usage.
- Keep agent-oriented authoring and validation instructions in this `AGENTS.md`.
- Stage and commit only files within the requested subject/documentation scope.
