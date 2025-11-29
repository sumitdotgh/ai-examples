import dataclasses
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from models import build_tiny_hope, build_tiny_transformer

# ==========================================================
# Color helper for pretty console output
# ==========================================================

class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


# ==========================================================
# Story-based continual learning curriculum
# ==========================================================

TEXT_CURRICULUM = {
    "Task_0": [
        "i pack my small bag for the train",
        "i walk to the station with my ticket",
        "i wait on the platform for the blue train",
        "i find my seat and watch trees go by",
    ],
    "Task_1": [
        "i pack my small bag for the flight",
        "i take a cab to the busy airport",
        "i wait in a long line at the gate",
        "i find my seat and watch clouds go by",
    ],
}


@dataclasses.dataclass
class TaskData:
    name: str
    inputs: np.ndarray
    targets: np.ndarray


# ==========================================================
# Vocabulary + token utilities
# ==========================================================

def build_vocab(curriculum: Dict[str, List[str]]) -> Dict[str, int]:
    vocab = {"<pad>": 0, "<eos>": 1}
    for sentences in curriculum.values():
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = len(vocab)
    return vocab


def sentence_to_arrays(sentence: str, vocab: Dict[str, int], seq_len: int):
    tokens = sentence.split() + ["<eos>"]
    tokens = tokens[:seq_len]
    tokens += ["<pad>"] * (seq_len - len(tokens))

    target_tokens = tokens[1:] + ["<pad>"]

    inputs = np.array([vocab[t] for t in tokens], dtype="int32")
    targets = np.array([vocab[t] for t in target_tokens], dtype="int32")
    return inputs, targets


def make_text_task(name, sentences, vocab, seq_len, repeats_per_sentence=128):
    inputs, targets = [], []
    for sentence in sentences:
        x, y = sentence_to_arrays(sentence, vocab, seq_len)
        for _ in range(repeats_per_sentence):
            inputs.append(x)
            targets.append(y)

    return TaskData(
        name=name,
        inputs=np.stack(inputs, axis=0),
        targets=np.stack(targets, axis=0),
    )


def create_curriculum(curriculum):
    vocab = build_vocab(curriculum)
    max_tokens = max(len(s.split()) for lst in curriculum.values() for s in lst) + 1

    tasks = [
        make_text_task(name, sentences, vocab, seq_len=max_tokens)
        for name, sentences in curriculum.items()
    ]
    return tasks, vocab, max_tokens


def as_dataset(task: TaskData, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((task.inputs, task.targets))
    ds = ds.shuffle(buffer_size=len(task.inputs), reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ==========================================================
# Continual learning core
# ==========================================================

def continual_train(model, tasks, *, epochs, batch_size):
    history = {}

    for task in tasks:
        print(f"\n{C.BOLD}{C.BLUE}📘 Training {model.name}{C.RESET}")
        print(f"   {C.CYAN}→ Starting {task.name}{C.RESET}")

        model.fit(as_dataset(task, batch_size=batch_size), epochs=epochs, verbose=0)

        print(f"   {C.GREEN}✓ Finished {task.name}{C.RESET}\n")

        checkpoint_metrics = {}
        for eval_task in tasks:
            ds = tf.data.Dataset.from_tensor_slices((eval_task.inputs, eval_task.targets))
            ds = ds.batch(batch_size)
            loss, acc = model.evaluate(ds, verbose=0)
            checkpoint_metrics[eval_task.name] = float(acc)

        history[task.name] = checkpoint_metrics

    return history


# ==========================================================
# Colored retention matrix
# ==========================================================

def print_history(name: str, history: Dict[str, Dict[str, float]]):
    tasks = list(history.keys())
    eval_tasks = list(history[tasks[0]].keys())

    print(f"\n{C.BOLD}{C.BLUE}==================== {name} Retention ===================={C.RESET}")
    print(
        f"{C.CYAN}Evaluation Task | "
        + " | ".join([f"After {t}" for t in tasks])
        + " | Forgetting"
        + C.RESET
    )
    print(C.YELLOW + "-" * 70 + C.RESET)

    for eval_task in eval_tasks:
        scores = [history[t][eval_task] for t in tasks]
        forgetting = scores[-1] - scores[0]
        forget_color = C.GREEN if forgetting >= 0 else C.RED

        row = (
            f"{C.BOLD}{eval_task:<15}{C.RESET}| "
            + " | ".join(f"{s:10.3f}" for s in scores)
            + f" | {forget_color}{forgetting:8.3f}{C.RESET}"
        )
        print(row)

    print(C.BLUE + "=" * 70 + C.RESET)


# ==========================================================
# Summary
# ==========================================================

def summarize(hist_a, hist_b):
    first_task = list(hist_a.keys())[0]
    final_task = list(hist_a.keys())[-1]

    transf_start = hist_a[first_task][first_task]
    transf_final = hist_a[final_task][first_task]
    hope_start = hist_b[first_task][first_task]
    hope_final = hist_b[final_task][first_task]

    t_forget = transf_final - transf_start
    h_forget = hope_final - hope_start

    print(f"\n{C.BOLD}{C.HEADER}📊 Continual Learning Summary{C.RESET}")
    print(C.YELLOW + "-" * 45 + C.RESET)

    t_color = C.GREEN if t_forget >= 0 else C.RED
    h_color = C.GREEN if h_forget >= 0 else C.RED

    print(f"- Transformer : {C.CYAN}{transf_final:.3f}{C.RESET}  "
          f"(forgot {t_color}{t_forget:.3f}{C.RESET})")
    print(f"- HOPE        : {C.CYAN}{hope_final:.3f}{C.RESET}  "
          f"(forgot {h_color}{h_forget:.3f}{C.RESET})")

    print("\n" + C.YELLOW + "👉 Interpretation:" + C.RESET)
    if transf_final > hope_final:
        print(f"{C.GREEN}Transformer retained more memory for this curriculum.{C.RESET}")
    elif hope_final > transf_final:
        print(f"{C.GREEN}HOPE retained more memory for this curriculum.{C.RESET}")
    else:
        print(f"{C.CYAN}Both models retained memory equally.{C.RESET}")

    print(C.YELLOW + "-" * 45 + C.RESET)


# ==========================================================
# Entrypoint
# ==========================================================

def main():
    np.random.seed(7)
    tf.random.set_seed(7)

    tasks, vocab, seq_len = create_curriculum(TEXT_CURRICULUM)
    vocab_size = len(vocab)

    transformer = build_tiny_transformer(vocab_size, seq_len)
    hope = build_tiny_hope(vocab_size, seq_len)

    transformer_history = continual_train(transformer, tasks, epochs=5, batch_size=64)
    hope_history = continual_train(hope, tasks, epochs=5, batch_size=64)

    print_history("Transformer", transformer_history)
    print_history("HOPE", hope_history)
    summarize(transformer_history, hope_history)


if __name__ == "__main__":
    main()
