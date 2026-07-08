# 🧠 Lingua Mentis — Dynamic Belief MNIST

> A neural network that accumulates perceptual bias in real time. Watch it drift, watch it be reset.

<div align="center">

**[Live Demo →](https://lingua-mentis.rajuan.app)**   ·   **[GitHub →](https://github.com/nadavrajuan/lingua-mentis-poc)**

</div>

---

## Demo

<div align="center">
  <a href="https://vimeo.com/share/0215210a-dd59-4352-ac37-1037de49939b?share=copy&fl=sv&fe=ci">
    <img src="https://img.shields.io/badge/▶_Watch_Demo-Vimeo-1ab7ea?style=for-the-badge&logo=vimeo&logoColor=white" alt="Watch Demo on Vimeo" height="40"/>
  </a>
</div>

---

## Motivation

Most machine learning systems are presented as stateless oracles. You feed an image in, a prediction comes out, and the system is identical to what it was before. This framing is mathematically convenient but philosophically strange — it describes something nothing in the natural world actually is.

Biological perception is fundamentally sequential. What you saw a moment ago changes how you interpret what you see now. A radiologist who just reviewed twenty ambiguous scans will read the twenty-first differently than if they had been looking at clear ones. A chess player mid-game is not processing each position cold. The brain is not a lookup table with a fixed key — it is a system whose present state is shaped by its recent history.

**Lingua Mentis** is an exploration of what it looks like to give a neural network a controlled version of this property.

It is not about making the model smarter. The base CNN is trained once on MNIST to 99.3% accuracy and then frozen. It is about giving the model a *short-term perceptual memory* — a mechanism by which recent experience subtly warps its next prediction. The drift is small, measurable, reversible, and analytically understood. And it makes the model's behavior in ways that are strangely human.

The name comes from Wittgenstein's *Lingua Franca* and the Latin *mentis* (of the mind). A shared language of error and correction.

---

## What It Does

At its core: a digit classifier that does not stay the same.

After each prediction, a small set of the model's weights are nudged — not the frozen base weights, but a separate *delta tensor* added on top of them. The nudge encodes the current image's internal representation and the model's confidence in its answer. Ambiguous images produce bigger nudges than clear ones.

The result is a classifier whose behavior shifts over sequences of predictions:

- Show it ten ambiguous `3/8` images, and the next image it sees — even a clear `4` — gets evaluated by a classifier that has been slightly pushed toward `3`.
- Enable return-to-normal, and the delta decays exponentially over time. Recent events weigh more than old ones.
- Reset the delta and the model snaps back to its base state instantly.

The probability chart shows both the frozen base model (*vanilla*) and the drifted model (*dynamic*) on every prediction. You can watch the gap between them open up.

---

## Architecture

```
                     INPUT IMAGE (28×28)
                           │
              ┌────────────▼────────────┐
              │  Conv2d(1→16, 3×3)      │
              │  ReLU + MaxPool(2×2)    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Conv2d(16→32, 3×3)     │
              │  ReLU + MaxPool(2×2)    │
              └────────────┬────────────┘
                           │
                        Flatten
                           │
              ┌────────────▼────────────┐
              │  Linear(800 → 128)      │  ← fc1
              │  ReLU                   │
              └────────────┬────────────┘
                           │
                    h ∈ ℝ¹²⁸   ← hidden representation
                     (frozen)
                           │
               ┌──────────┴──────────┐
               │                     │
      ┌────────▼───────┐   ┌─────────▼────────┐
      │  base_fc2_w    │   │  base_fc2_w       │
      │  (10×128)      │   │  + delta_fc2_w    │
      │  base_fc2_b    │   │  (10×128)         │
      │  (10,)         │   │  base_fc2_b       │
      │  FROZEN        │   │  + delta_fc2_b    │
      └────────┬───────┘   └─────────┬────────┘
               │                     │
       vanilla logits         dynamic logits
               │                     │
           softmax               softmax
               │                     │
      vanilla probs           dynamic probs
      (gray ghost bar)       (colored bar)
```

The key architectural choice: the delta weights live in a separate tensor, initialized to zero. The effective classifier is `base + delta`. Vanilla and dynamic probabilities both derive from the same hidden vector `h` — they diverge only at the final layer.

---

## The Update Rule

After each prediction, `delta_fc2_weight` is updated using a pseudo-Hebbian rule:

```
h      = hidden activation for the current image  (shape 128)
p      = dynamic softmax probabilities            (shape 10)
ŷ      = predicted class = argmax(p)
e      = one-hot vector for ŷ                    (shape 10)

direction        = e − λ·p            (λ = inhibition_strength, default 1.0)
direction[ŷ]     = 1 − p[ŷ]          (override: push predicted class up)

h_norm           = h / ‖h‖

update_matrix    = outer(direction, h_norm)   →  shape (10, 128)

scale            = intensity × (1 − max(p))  ←  confidence factor
delta_fc2_weight += scale × update_matrix
delta_fc2_bias   += scale × direction
```

The confidence factor `(1 − max(p))` is the key: a 97% confident prediction moves the weights by 3% of the base update magnitude; a 60% confident prediction moves them by 40%. **Ambiguous images drive stronger drift than clear ones.** This mirrors the biological principle that surprising events create stronger memory traces.

```
  Confidence     Factor    Effect
  ──────────    ────────   ──────────────────────────────────────────
    99%           0.01     Barely a whisper. Clear images leave barely a trace.
    80%           0.20     Moderate nudge. The model noticed something.
    60%           0.40     Significant update. Ambiguity is costly.
    40%           0.60     Large update. The model is genuinely uncertain.
```

---

## Exponential Decay (Return to Normal)

When enabled, the delta decays between predictions:

```
delta_fc2_weight *= exp(−return_rate × Δt)
delta_fc2_bias   *= exp(−return_rate × Δt)
```

This gives the model a fading memory — older events matter less than recent ones, exactly like exponential weighting in time series. The drift chart in the UI shows delta norm over the last 120 steps: you can watch it rise during ambiguous sequences, decay during pauses, and collapse on manual reset.

```
  return_rate    Half-life
  ───────────    ─────────
     0.0           ∞        (no decay — drift accumulates forever)
     0.05        ~14 s
     0.20         ~3.5 s
     0.50         ~1.4 s
     2.00        ~0.35 s
```

---

## Reading the Probability Chart

Each column represents one digit class (0–9):

```
  100% ┤
       │
       │     ██
       │     ██ ▓▓
       │  ██ ██ ▓▓       ██
   50% ┤  ██ ██ ▓▓       ██
       │  ██ ██ ▓▓  ░░   ██   ░░
       │  ██ ██ ▓▓  ░░   ██   ░░
    0% └──────────────────────────
           0  1  2  3  4  5 ...
```

- **Gray / outline** — vanilla (frozen base model) probability
- **White horizontal line** — marks where the vanilla probability tops out on that column
- **Green** — dynamic probability is *above* vanilla (this class has been boosted by drift)
- **Red** — dynamic probability is *below* vanilla (this class has been suppressed)
- **Purple** — the predicted class (argmax of dynamic probabilities)
- **+/− label above** — the exact delta in percentage points

As drift accumulates, you see the colored bars drift away from their white lines. After a run of ambiguous `3` vs `8` images, class 3 and 8 bars will show pronounced divergence from their vanilla levels even on completely unrelated digits.

---

## The Deeper Point: What This Is Not

This is not training. The base model does not improve at MNIST. It cannot unlearn the clean weight matrix.

What this is: a demonstration that a fixed, high-accuracy inference function can be made to exhibit *stateful, sequential, directional perceptual bias* — through a controlled, reversible, analytically understood mechanism in the final layer — without changing a single base weight.

The delta lives on top of the model the way attention or working memory might live on top of long-term representation. It is not stored in the image. It is not a cache. It is in the weights — specifically, in 1,290 numbers that mediate between the model's 128-dimensional internal representation of the world and its 10-class decision about what it is looking at.

When you reset the delta, you are not erasing experience. You are resetting *recency*. The base model remembers everything it was trained on. The delta only holds what happened in the last few seconds.

The calibration feature extends this idea: if a human tags a sequence of ambiguous images, the system can find the dynamic parameter set that would have made the model's probability distributions closest to the human's choices. Not the same accuracy — the same *pattern of uncertainty* over the same images.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, WebSocket |
| ML | PyTorch CNN (CPU-only, ~500MB image) |
| Data | MNIST test set (10,000 images), SQLite |
| Frontend | Vanilla JS, CSS animations, Canvas API |
| Infra | Docker, Traefik, EC2 (t3.small) |
| CI/CD | GitHub Actions → GHCR → EC2 SSH deploy |

---

## Running Locally

```bash
git clone https://github.com/nadavrajuan/lingua-mentis-poc
cd lingua-mentis-poc

docker compose up
```

On first boot the container will:
1. Download the MNIST dataset (~12MB)
2. Train the base CNN for 5 epochs on CPU (~2–3 minutes)
3. Build an ambiguity bank from the 10,000 test images
4. Start the FastAPI server on `http://localhost:8000`

No GPU required. No API keys. No accounts.

---

## Controls

| Control | Default | What it does |
|---|---|---|
| **Intensity** | 0.01 | How much each prediction moves the delta weights |
| **Max Δ norm** | 3.0 | Hard ceiling on total weight drift (Frobenius norm) |
| **Confidence curve** | inverse | How confidence maps to update magnitude |
| **Return to normal** | Off | Exponential decay of delta between predictions |
| **Return rate** | 0.2 | Decay speed (inverse seconds) |
| **Noise σ** | 0.0 | Gaussian noise added to each image before inference |
| **Sequence mode** | Random | Which MNIST images to show (random, ambiguous, class-specific) |
| **Interval (ms)** | 700 | Time between automatic predictions in playback mode |
| **Show true label** | On | Show ground-truth label from test set |

For detailed parameter documentation, mathematical derivations, and calibration methodology, see [SYSTEM.md](./SYSTEM.md).

---

## Project Structure

```
lingua-mentis-poc/
├── app/
│   ├── model.py          # MNISTCNN architecture + load_model
│   ├── dynamic_engine.py # Delta weights, update rule, decay, predict()
│   ├── schemas.py        # Pydantic models (DynamicParams, PredictionResult)
│   ├── sampler.py        # Image sampling from ambiguity bank
│   ├── data.py           # MNIST dataset loading + image rendering
│   ├── calibration.py    # Grid search to fit params to human sessions
│   ├── human_training.py # Human trial session management
│   ├── db.py             # SQLite async persistence (aiosqlite)
│   ├── main.py           # FastAPI app, WebSocket playback loop
│   └── static/           # index.html, app.js, style.css
├── scripts/
│   ├── bootstrap.py          # First-boot: download, train, build bank
│   ├── train_base_model.py   # CNN training loop
│   └── build_ambiguity_bank.py # Confidence + margin scoring of test set
├── SYSTEM.md             # Full technical + conceptual documentation
├── Dockerfile
├── docker-compose.yml
└── docker-compose.prod.yml
```

---

<div align="center">
  <sub>Built by <a href="https://github.com/nadavrajuan">nadavrajuan</a> · <a href="https://lingua-mentis.rajuan.app">lingua-mentis.rajuan.app</a></sub>
</div>
