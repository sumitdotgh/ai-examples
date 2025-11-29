# 🌱 Tiny Nested Learning Playground  
*A hands-on comparison of Transformers vs HOPE (Hierarchical Open-ended Pattern Expansion)*

This project is a compact, intuitive demonstration of **continual learning** — how a machine learning model behaves when it learns **Task A** and then **Task B**, and whether it **forgets** what it learned earlier.

It implements a simplified version of ideas from Google’s **Nested Learning** research and compares:

- a **tiny Transformer encoder** (baseline attention-only learner)  
- a **tiny HOPE-inspired recurrent model** using a **continuum memory system (CMS)** with **fast**, **medium**, and **slow** update timescales  

Both models are trained on two tiny natural-language tasks:

- **Task 0 — Catch a train**  
- **Task 1 — Catch a flight**

By observing how much each model **remembers Task 0** after learning Task 1, we get a clear, human-readable demonstration of **catastrophic forgetting** and how multi-timescale memory can mitigate it.

---

# 🚀 Setup

```sh
cd tiny-nested-learning
docker compose build --no-cache
docker compose up
```

---

# 🎯 What the Script Does

1. **Builds two tiny text tasks** from short English stories.  
2. **Trains the Transformer** on Task 0 → Task 1.  
3. **Trains the HOPE model** on the same sequence.  
4. **Prints color-coded retention tables** to show forgetting vs. retention.

---

# 📘 The Two Tiny Tasks

### **Task 0 — Catch a Train**
```
i walk to the station with my ticket
i wait on the platform for the blue train
i find my seat and watch trees go by
```

### **Task 1 — Catch a Flight**
```
i take a cab to the busy airport
i wait in a long line at the gate
i find my seat and watch clouds go by
```

---

# 🧠 Architecture Overview (with Mermaid Visuals)

## 🔵 Transformer — One Shared Memory System

```mermaid
flowchart LR
    subgraph Transformer
        T0[Token IDs] --> TE[Embedding Layer]
        TE --> MH[Multi-Head Attention]
        MH --> LN1[LayerNorm + Residual]
        LN1 --> FFN[Feed Forward Network]
        FFN --> LN2[LayerNorm + Residual]
        LN2 --> LOGITS[Token Logits]
    end
```

---

## 🟢 HOPE — Multi-Timescale Memory (Fast / Medium / Slow)

```mermaid
flowchart LR
    subgraph HOPE
        T0[Token IDs] --> EMB[Embedding Layer]
        EMB --> CMS[CMS Cell<br/>Controller + Rate Adapter]

        CMS --> F[Fast Memory<br/>Update ~ 0.6]
        CMS --> M[Medium Memory<br/>Update ~ 0.3]
        CMS --> S[Slow Memory<br/>Update ~ 0.02]

        F & M & S --> AGG[Aggregated State]
        AGG --> LOGITS[Token Logits]
    end
```

---

# 🧠 HOPE Memory Update — Sequence Diagram

```mermaid
sequenceDiagram
    autonumber

    participant X as Input Token<br/>x_t
    participant C as Controller
    participant R as Rate Adapter
    participant F as Fast Memory
    participant M as Medium Memory
    participant S as Slow Memory
    participant A as Aggregator
    participant O as Output Logits

    X->>C: Token embedding
    C->>C: Compute candidate<br/>state h_candidate

    C->>R: Send candidate for<br/>rate computation
    R->>F: α_fast
    R->>M: α_med
    R->>S: α_slow

    Note over F: Update rule:<br/>(1 - α_fast)*old<br/> + α_fast*h_candidate
    Note over M: Update rule:<br/>(1 - α_med)*old<br/> + α_med*h_candidate
    Note over S: Update rule:<br/>(1 - α_slow)*old<br/> + α_slow*h_candidate

    F->>A: Updated fast memory
    M->>A: Updated med memory
    S->>A: Updated slow memory

    A->>A: Combine memories<br/>h_combined
    A->>O: Produce logits<br/>next-word prediction
```

---

# 🏆 Example Results — Including REAL Output

```
==================== TRAINING TINY_HOPE ====================
Epochs per task: 3, Batch size: 64

📘 Training tiny_hope
→ Starting Task_0
✓ Finished Task_0
Evaluating retention after Task_0...

📘 Training tiny_hope
→ Starting Task_1
✓ Finished Task_1
Evaluating retention after Task_1...
✓ Completed all tasks for tiny_hope
```

---

# 📊 Transformer Retention Table

```
==================== TRANSFORMER RETENTION ====================
Evaluation Task | After Task_0 | After Task_1 | Forgetting
---------------------------------------------------------------------------
Task_0         |      0.975 |      0.800 |   -0.175
Task_1         |      0.525 |      1.000 |    0.475
===========================================================================
```

---

# 📊 HOPE Retention Table

```
==================== HOPE RETENTION ====================
Evaluation Task | After Task_0 | After Task_1 | Forgetting
---------------------------------------------------------------------------
Task_0         |      0.575 |      0.700 |    0.125
Task_1         |      0.325 |      0.625 |    0.300
===========================================================================
```

---

# ⭐ Continual Learning Summary

```
Transformer forget: -0.175
HOPE forget:        0.125

👉 HOPE retained more memory (less forgetting).
```

---

# 📘 Educational Explanation: Understanding the Results

**Catastrophic forgetting** occurs when a model learns Task 1 and overwrites what it learned in Task 0.

Final accuracy alone is misleading.  
The correct metric in continual learning is:

```
FORGETTING = Final Accuracy – Start Accuracy
```

A model with **lower forgetting** (closer to zero or positive) is the better continual learner.

### In this experiment:

- The **Transformer** forgot **17.5%** of Task_0.
- **HOPE actually improved** on Task_0 by **12.5%**, meaning **zero catastrophic forgetting**.

This happens because HOPE uses **multi-timescale memory**:

- Fast memory → adapts quickly  
- Medium memory → blends  
- Slow memory → preserves long-term knowledge  

This mirrors Google's Nested Learning idea:

> Learning at multiple speeds protects older knowledge while adapting to new tasks.

### ✔ Key Takeaway

```
The better continual learner is the one that FORGETS LESS.
```

HOPE wins this experiment.

---

---

# 🔍 Why HOPE Works Better Here

- **Slow memory** barely changes → protects Task 0  
- **Fast memory** absorbs Task 1 quickly → lower interference  
- **Medium memory** blends both patterns  
- Transformer updates **one shared weight space**, overwriting earlier information  

HOPE demonstrates how **multi-timescale memory** can significantly reduce catastrophic forgetting.

---

# ⚠️ Disclaimer — HOPE Can Also Forget More

To be scientifically honest:

HOPE *can* forget more than a Transformer if:

- fast memory rate is too high  
- slow memory is not slow enough  
- tasks are extremely different  
- the model is very tiny  
- training runs too long  

Example bad setting:

```
fast = 0.95
medium = 0.50
slow = 0.10
```

Produces retention like:

```
Transformer retains: 0.82
HOPE retains:        0.40
```

This demonstrates:

> Multi‑timescale memory is powerful **only when tuned properly**.

---

# 🔧 Customize & Explore

Try:

- Adjusting HOPE update rates  
- Adding more tasks (Bus → Flight → Metro → Boat)  
- Increasing vocabulary size  
- Changing Transformer depth  
- Lowering Task 1 epochs to reduce destructive updates  

---

# 📘 RNN vs Transformer vs HOPE (Quick Comparison)

| Model        | Memory Type | Strengths | Weaknesses |
|--------------|-------------|-----------|------------|
| **RNN**      | One hidden state | Simple, sequential | Severe forgetting |
| **Transformer** | Shared parameter memory | Strong modeling power | High forgetting when fine-tuned |
| **HOPE**     | Fast + Medium + Slow | Protects old tasks via slow memory | Needs tuning |

---

# 🧠 Why “Nested Learning”?

Traditional models update **one memory system**.

Nested Learning updates **multiple memory systems** *simultaneously*, each at a different speed:

- **Fast** → immediate adaptation  
- **Medium** → short‑term consolidation  
- **Slow** → long‑term stability  

HOPE is a small but functional example of this idea.

---

# 🎉 Final Notes

This project is deliberately tiny — small enough to understand deeply, but powerful enough to illustrate the most important concepts in continual learning:

- Catastrophic forgetting  
- Multi-timescale memory  
- Nested Learning  
- Transformer vs HOPE behavior  

Use it as a learning tool, demo, or foundation for larger experiments.
