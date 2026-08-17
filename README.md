# PyTorch Mastery Through Exercises

A personal collection of progressive PyTorch exercise notebooks.

## Goal

The goal of this repository is to gain experience through **coding and mental exercises**, not passive reading. Each notebook asks you to predict shapes, values, axes, or gradients before using PyTorch to verify your reasoning. Repetition is intentional: the paired virgin notebooks make it easy to retry a subject from a clean state after enough time has passed.

## Exercise notebooks

| Subject | Working notebook | Clean retry notebook | Scope |
|---|---|---|---:|
| Tensor indexing | [`torch_indexing_exercises.ipynb`](notebooks/torch_indexing_exercises.ipynb) | [`torch_indexing_exercises_virgin.ipynb`](notebooks/torch_indexing_exercises_virgin.ipynb) | 110 exercises |
| Broadcasting | [`torch_broadcasting_exercises.ipynb`](notebooks/torch_broadcasting_exercises.ipynb) | [`torch_broadcasting_exercises_virgin.ipynb`](notebooks/torch_broadcasting_exercises_virgin.ipynb) | 110 exercises |
| Tensor shapes and manual backpropagation | [`tensor_backprop_exercises.ipynb`](notebooks/tensor_backprop_exercises.ipynb) | [`tensor_backprop_exercises_virgin.ipynb`](notebooks/tensor_backprop_exercises_virgin.ipynb) | 155 exercises |

Files beginning with `_` in `notebooks/` provide private runtime references for tests. Keep them beside the corresponding notebooks.

## Working and virgin copies

Every subject has two notebook files:

- `*_exercises.ipynb` is the **working copy**. It may contain answers, experiments, execution history, and scratch work.
- `*_exercises_virgin.ipynb` is the **pristine retry copy**. Its learner-answer cells contain comments only, and saved execution outputs are cleared.

Do not work directly in a virgin notebook. To retry without overwriting current progress, copy it first:

```bash
cp notebooks/torch_broadcasting_exercises_virgin.ipynb \
   notebooks/torch_broadcasting_retry.ipynb
```

To reset the canonical working copy intentionally:

```bash
cp notebooks/torch_broadcasting_exercises_virgin.ipynb \
   notebooks/torch_broadcasting_exercises.ipynb
```

The second command destroys answers in the working copy, so commit or back it up first.

## Setup

This repository uses a small `pyproject.toml` so it can be opened with `uv`:

```bash
uv sync
uv run jupyter lab
```

Then open a notebook and run its setup/helper cell before Exercise 001.

## Recommended practice loop

For each exercise:

1. Read the purpose and name the meaning of every axis.
2. Predict required shapes, aligned shapes, values, or gradients on paper or in the answer cell.
3. Narrate the operation mentally before executing it.
4. Write the smallest clear PyTorch expression.
5. Run the supplied private-reference test.
6. If it fails, explain why before changing code.
7. Move on only when the result passes and the reasoning can be stated aloud.

When stuck, ask for one hint about the current reasoning step rather than requesting the complete answer.

## Creating an exercise notebook for a new subject

Use this process for future topics such as tensor views, reductions, matrix multiplication, autograd, convolution shapes, attention, or numerical stability.

### 1. Define the mastery target

Write one sentence describing what repeated coding and mental practice should make automatic. Break the topic into a progression from concrete one-operation cases to realistic machine-learning patterns and capstones.

A useful progression is:

1. vocabulary and rank-zero/rank-one foundations;
2. one concept at a time with tiny visible tensors;
3. shape prediction before execution;
4. common misconceptions and deliberately invalid cases;
5. higher-rank combinations;
6. storage or autograd semantics when relevant;
7. realistic PyTorch patterns;
8. integrated capstones.

### 2. Follow the four-cell exercise contract

Give every exercise stable IDs and place these cells in order:

1. `exercise-NNN-prompt` — Markdown describing purpose, visible inputs, task, ingredients, required outputs, and next concept.
2. `exercise-NNN-fixture` — supplied input construction and private-reference registration; it must not print shape or value answers.
3. `exercise-NNN-answer` — comments and blank space for learner code only.
4. `exercise-NNN-test` — checks required variables without embedding expected tuples or tensors in visible test source.

Keep central mathematical relationships in standalone display LaTeX, define every symbol in prose, and keep explanations close to the code they describe.

### 3. Make tests useful without giving answers away

Tests should check type, shape, dtype, and values while reporting only actionable feedback such as “reconsider right alignment.” Do not write visible assertions like:

```python
_check_shape("result_shape", (2, 3))
```

Instead, derive expected results at runtime in a generic helper or a private support module such as `_topic_fixtures.py`, then let the test refer to a key and field.

Visible fixtures should expose the learner’s inputs and forward operation, but only expected answers should remain private.

### 4. Create the working notebook

Name it:

```text
notebooks/<subject>_exercises.ipynb
```

Include:

- a title and usage instructions;
- a mental checklist;
- a course map with exercise ranges;
- setup and private test helpers;
- progressive prompt/fixture/answer/test groups;
- a completion standard and official references.

### 5. Generate the virgin pair

First remove accidental scratch cells that should not be part of the reusable curriculum. Then run:

```bash
python scripts/create_virgin.py notebooks/<subject>_exercises.ipynb
```

The script creates:

```text
notebooks/<subject>_exercises_virgin.ipynb
```

It blanks cells whose IDs match `exercise-NNN-answer`, clears all code-cell outputs and execution counts, and verifies that no executable answer code remains. Custom notebooks should therefore preserve the standard answer-cell IDs.

### 6. Validate before adding the subject

Check all of the following:

- notebook JSON parses successfully;
- every cell ID is unique;
- every code cell compiles;
- setup and all visible fixtures run sequentially;
- private references cover every required test field;
- simulated correct answers pass every test;
- virgin answer cells contain no executable statements;
- virgin notebooks contain no saved outputs;
- tests reveal no expected shape tuples or answer tensors;
- documentation matches executable operations and sample counts.

Finally, add the new working/virgin pair to the table in this README.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── notebooks/
│   ├── *_exercises.ipynb
│   ├── *_exercises_virgin.ipynb
│   └── _*_fixtures.py
└── scripts/
    └── create_virgin.py
```
