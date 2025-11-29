# Tiny Nested Learning Playground

This example mirrors the Google Nested Learning blog post by contrasting a tiny Transformer baseline with a simplified HOPE-style recurrent block that keeps memories at multiple time scales. It is intentionally small so you can understand the mechanics.

## Environment

TensorFlow 2.15 only supports Python 3.11, so make sure you point Poetry at the right interpreter:

```sh
cd /Users/sumitghosh/source/github/ai-examples/tiny-nested-learning
poetry env use /usr/local/bin/python3.11
poetry install
poetry shell
```

## Running the experiment

```sh
python main.py
```

The script will:

1. Build two natural-language tasks (Pets/Home vs Weather/Trips) using short English sentences.
2. Train both the tiny Transformer (`models/transformer.py`) and the HOPE-style model (`models/hope.py`) task-by-task.
3. Report token-level accuracy after each task so you can observe forgetting and retention.
4. Emit a short textual comparison summary.

Adjust hyper-parameters at the bottom of `main.py` to explore different memory scales, Transformer width, or curriculum schedules.


