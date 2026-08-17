# PyTorch Mastery Through Exercises

A personal collection of progressive PyTorch exercise notebooks.

## Goal

The goal of this repository is to gain experience through **coding and mental exercises**, not passive reading. Each notebook asks you to predict shapes, values, axes, or gradients before using PyTorch to verify your reasoning. Repetition is intentional: the paired virgin notebooks make it easy to retry a subject from a clean state after enough time has passed.

## Exercise notebooks

| Subject | Working notebook | Clean retry notebook | Scope |
|---|---|---|---:|
| Tensor indexing | [`torch_indexing_exercises.ipynb`](notebooks/torch_indexing_exercises.ipynb) | [`torch_indexing_exercises_virgin.ipynb`](notebooks/torch_indexing_exercises_virgin.ipynb) | 110 exercises |
| Broadcasting | [`torch_broadcasting_exercises.ipynb`](notebooks/torch_broadcasting_exercises.ipynb) | [`torch_broadcasting_exercises_virgin.ipynb`](notebooks/torch_broadcasting_exercises_virgin.ipynb) | 64 exercises |
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
2. Predict only the shapes, values, alignment details, or gradients requested by the current exercise.
3. Narrate the operation mentally before executing it.
4. Write the smallest clear PyTorch expression.
5. Run the supplied private-reference test.
6. If it fails, explain why before changing code.
7. Move on only when the result passes and the reasoning can be stated aloud.

When stuck, ask for one hint about the current reasoning step rather than requesting the complete answer.

## Repository layout

```text
.
├── README.md
├── AGENTS.md
├── pyproject.toml
├── notebooks/
│   ├── *_exercises.ipynb
│   ├── *_exercises_virgin.ipynb
│   └── _*_fixtures.py
└── scripts/
    └── create_virgin.py
```
