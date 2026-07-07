# Lingua Mentis — System Documentation

## What is this?

A simple MNIST digit classifier that does not stay fully static during prediction. After each prediction, the system applies a small, temporary change to a subset of the model's weights — based on what the model just saw and how confident it was. The next prediction is made by a slightly different model than the last.

The result is a machine that has a short-term perceptual memory. It can become biased. It can drift. It can be reset. And it can be calibrated to behave more like a specific human.

---

## Part 1: Under the Hood (Technical)

### The Model Architecture

A small convolutional neural network trained on MNIST once, then frozen.

```
Input image: 28×28 grayscale pixel array

Layer 1:  Conv2d(1 → 16 channels, kernel 3×3)   → ReLU → MaxPool2d(2)
Layer 2:  Conv2d(16 → 32 channels, kernel 3×3)  → ReLU → MaxPool2d(2)
Flatten:  32 × 5 × 5 = 800 values
Layer 3:  Linear(800 → 128)   → ReLU        ← hidden vector h ∈ ℝ¹²⁸
Layer 4:  Linear(128 → 10)                  ← logits → softmax → p ∈ ℝ¹⁰
```

The hidden vector **h** is the compressed representation of the input image. Everything before it is a feature extractor. Layer 4 (called `fc2`) is the classifier head — it maps that 128-dimensional representation to 10 class scores.

### What Gets Updated

The base model weights are **never touched**. Instead, the system maintains a parallel delta tensor:

```
base_fc2_weight:   shape (10, 128)   — frozen
base_fc2_bias:     shape (10,)       — frozen

delta_fc2_weight:  shape (10, 128)   — updated each step
delta_fc2_bias:    shape (10,)       — updated each step
```

Total weights in fc2: `10 × 128 + 10 = 1,290 parameters`.

The effective classifier used for each prediction is:

```
effective_weight = base_fc2_weight + delta_fc2_weight
effective_bias   = base_fc2_bias   + delta_fc2_bias

logits = effective_weight @ h + effective_bias
p      = softmax(logits)
```

This is what the probability chart calls **dynamic** probabilities. The **vanilla** probabilities shown in gray are what the frozen base model would output on the same hidden vector.

### The Update Rule

After each prediction, `delta_fc2_weight` and `delta_fc2_bias` are adjusted using a pseudo-Hebbian / gradient-like rule.

Let:
- `h` = hidden activation vector for the current image (shape 128)
- `p` = softmax output of the dynamic model (shape 10)
- `ŷ` = predicted class (argmax of p)
- `e` = one-hot vector for ŷ (shape 10)

The update direction is:

```
direction = e − λ · p
direction[ŷ] = 1 − p[ŷ]          (override predicted class slot)
```

where `λ` = `inhibition_strength`. With λ=1 this becomes the standard cross-entropy gradient direction for the predicted label.

The outer product of `direction` with the normalized hidden vector gives the weight update:

```
h_norm = h / ||h||

update_matrix = outer(direction, h_norm)    shape: (10, 128)
update_bias   = direction                   shape: (10,)
```

This is then scaled by intensity and a confidence factor:

```
delta_fc2_weight += scale · update_matrix
delta_fc2_bias   += scale · update_bias

scale = intensity × confidence_factor
```

The key property: this update **strengthens the path from the current hidden pattern toward the predicted class**, while **weakening paths to competing classes**. Over repeated exposure to similar images, the model becomes biased toward whichever label it has been predicting.

### Confidence Factor

Ambiguous predictions produce larger updates than confident ones. The default mode:

```
confidence_factor = 1 − max(p)
```

So a 97% confident prediction changes the weights by only 3% of the raw update magnitude. A 60% confident prediction changes them by 40%.

| Confidence | confidence_factor | Effective scale (intensity=0.01) |
|-----------|-------------------|----------------------------------|
| 0.99      | 0.01              | 0.0001                           |
| 0.80      | 0.20              | 0.002                            |
| 0.60      | 0.40              | 0.004                            |
| 0.40      | 0.60              | 0.006                            |

This is the mechanism by which **ambiguous images drive stronger drift** than clear ones. A clean `1` barely moves anything. A fuzzy `3/8` shifts the model noticeably.

### Norm Clamping (Safety)

Two limits prevent numerical explosion:

**Per-update clamp** — before adding, the update is scaled down if it would exceed `max_single_update_norm`:

```python
if ||update_matrix|| × scale > max_single_update_norm:
    scale = scale × max_single_update_norm / (||update_matrix|| × scale)
```

**Global delta clamp** — after adding, if the total delta norm exceeds `max_delta_norm`, the entire delta is rescaled:

```python
if ||delta_fc2_weight|| + ||delta_fc2_bias|| > max_delta_norm:
    delta_fc2_weight *= max_delta_norm / current_norm
    delta_fc2_bias   *= max_delta_norm / current_norm
```

The delta norm displayed in the UI is `||delta_fc2_weight|| + ||delta_fc2_bias||`. It grows during drift and shrinks during decay.

### Return-to-Normal Decay

Between predictions, the delta decays exponentially:

```
Δt = time elapsed since last prediction

delta_fc2_weight *= exp(−return_rate × Δt)
delta_fc2_bias   *= exp(−return_rate × Δt)
```

| return_rate | Half-life (time for delta to halve) |
|-------------|--------------------------------------|
| 0.0         | ∞ (no decay)                         |
| 0.05        | ~14 seconds                          |
| 0.2         | ~3.5 seconds                         |
| 0.5         | ~1.4 seconds                         |
| 2.0         | ~0.35 seconds                        |

The decay is applied as part of each prediction call — it accounts for however long has elapsed since the last one.

### Calibration

The calibration feature fits dynamic parameters to a human tagging session using grid search.

Given a sequence of (image, human_label) pairs, for each candidate parameter set:

1. Reset delta to zero.
2. Replay the exact same image sequence.
3. For each image, compute model probabilities under the current delta state.
4. Measure how much probability mass the model assigned to the human's answer.

Loss function:

```
loss = mean(−log p[human_label])  +  stability_penalty
```

This is standard negative log-likelihood: it is minimized when the model's probability distribution over each image matches where the human actually clicked.

Grid searched over 150 combinations:

```
intensity          × [0.001, 0.003, 0.01, 0.03, 0.1]
return_rate        × [0.0, 0.05, 0.2, 0.5, 1.0]
inhibition         × [0.5, 1.0, 1.5]
confidence_mode    × [inverse_confidence, binned]
```

---

## Part 2: Simpler Explanation (Still Technical)

### The basic idea

A neural network classifier is just a function that maps an input image to a probability distribution over classes. At inference time — when it's making predictions — that function is defined entirely by its weights.

This system makes a small modification to a small part of those weights after each prediction.

Think of the final layer of the network as a **scoring matrix**: it has one row per class (10 rows for digits 0–9), and each row is a vector of 128 numbers that says "how much does each position in the hidden layer contribute to this class's score?" The model predicts whichever class has the highest score.

When we update `delta_fc2_weight`, we're nudging those 10 rows. If the model just predicted `3`, we push row 3 slightly upward in the directions activated by the current hidden pattern, and push the other rows slightly downward. The next time the model sees a similar image (one that activates a similar hidden pattern), row 3 will score a little higher.

### Why the delta lives in a separate tensor

We never modify `base_fc2_weight` directly. Instead we keep a separate `delta_fc2_weight` initialized to zero and add them together before each forward pass.

This is important for two reasons:

- **Resettability**: hitting Reset just zeros out the delta. The base model is untouched.
- **Comparability**: we can run the same image through both `base + 0` (vanilla) and `base + delta` (dynamic) and compare. That's where the vanilla/dynamic split in the probability chart comes from.

### What the probability chart actually shows

Each bar represents one digit class (0–9).

- The **gray ghost bar** is the base model's probability for that class — what it would say if no drift had accumulated.
- The **colored foreground bar** is the dynamic model's probability — after factoring in all the weight nudges from previous predictions.
- A **white horizontal line** on each bar marks where the ghost bar tops out. If the colored bar is above the line, that class has been boosted. Below, it's been suppressed.
- **Green** = boosted above vanilla. **Red** = suppressed below vanilla. **Purple** = the predicted class. The delta label (e.g., `+8.3%`) shows the exact shift.

As you watch the playback, you should see the bars animate. After several predictions of `3`, the class 3 bar in the next prediction will tend to sit higher than its ghost line — even on a different image. That's the drift made visible.

### Why ambiguity matters more

The update magnitude is proportional to `1 − confidence`. A confident prediction means the model's distribution is already peaked — the one-hot target and the softmax output are close, so the gradient is small. An uncertain prediction means they're far apart — the gradient is large.

This is not an arbitrary design choice. It mirrors what happens in biological learning: surprising or ambiguous events create stronger memory traces than expected, unsurprising ones.

In practice: drop a clear `7` into the model at 99% confidence and the delta barely moves. Show it a blurry `3/8` at 55% confidence and the weights shift noticeably. Then the next digit — even a totally different one — gets evaluated by a slightly warped classifier.

### The "session memory" model

Think of each session as a short-term context window, but stored in the weights rather than in activations.

What the model has "seen" recently is encoded implicitly in `delta_fc2_weight`. It doesn't store the images. It stores the cumulative effect of how those images moved the scoring matrix. The delta norm is a single number measuring how far the model has drifted from its neutral state.

When `return_to_normal` is enabled, the delta decays over time. This makes the model's memory exponentially weighted toward the recent past — older predictions matter less the further back they are.

---

## Parameter Reference

### Intensity

**What it does:** Controls how much each prediction moves the weights.

**Range:** 0.001 (barely any drift) to 0.2 (aggressive drift)

**Default:** 0.01

Setting this too high causes rapid drift and instability — the model can lock onto a class and stop classifying other digits correctly. Setting it too low means you need many predictions before any drift is visible.

---

### Interval (ms)

**What it does:** Time between automatic predictions during playback.

**Default:** 700ms

Only affects the playback timer. Shorter intervals mean faster drift accumulation per unit of real time.

---

### Return to Normal

**What it does:** When enabled, the delta weights decay exponentially toward zero between predictions.

**Default:** Off

Turning this off lets drift accumulate indefinitely (capped only by `max_delta_norm`). Turning it on gives the model a fading memory — recent predictions matter more than older ones.

---

### Return Rate

**What it does:** How fast the delta decays when return-to-normal is on.

**Unit:** inverse seconds (rate constant of exponential decay)

**Range:** 0.0 (no decay) to 2.0 (very fast decay)

**Half-lives:** 0.05 → ~14s, 0.2 → ~3.5s, 0.5 → ~1.4s, 2.0 → ~0.35s

---

### Confidence Curve

**What it does:** Determines how confidence maps to update magnitude.

| Mode | Description |
|------|-------------|
| `inverse_confidence` | `factor = 1 − conf`. Smooth linear penalty for high confidence. Default. |
| `binned` | Step function: ≥90% → 0.15, ≥75% → 0.40, ≥60% → 0.75, else → 1.0 |
| `margin_based` | Same formula as inverse_confidence (margin version planned) |
| `fixed` | Always 1.0. Updates are the same size regardless of confidence. |

---

### Dynamic Layer

**What it does:** Selects which layers of the network receive delta updates.

| Mode | Layers updated | Parameters changed |
|------|----------------|--------------------|
| `fc2 only` | Final classifier | 1,290 |
| `fc1 + fc2` | Hidden layer + classifier | 103,562 |

`fc2 only` is faster, more stable, and more interpretable. `fc1 + fc2` allows the model's feature representations themselves to drift, which can produce stronger and stranger effects.

---

### Max Δ Norm

**What it does:** Hard cap on the total magnitude of accumulated weight changes.

**Default:** 3.0

This is the Frobenius norm of `delta_fc2_weight` plus the L2 norm of `delta_fc2_bias`. If drift would exceed this, the entire delta tensor is rescaled proportionally. Acts as a stability ceiling.

---

### Image Noise σ

**What it does:** Adds independent Gaussian noise to each image before display and before the model sees it.

**Unit:** Standard deviations in the normalized pixel space (MNIST is normalized to mean≈0.13, std≈0.31)

**Range:** 0.0 (clean) to 1.5 (heavily corrupted)

Noise is re-randomized each prediction — the same digit looks different every time it appears. Higher noise lowers confidence, which means larger weight updates per image and faster drift accumulation.

---

### Sequence Mode

**What it does:** Controls which MNIST test images are shown.

| Mode | Description |
|------|-------------|
| Random | All 10,000 test images, uniformly sampled |
| Only digit A / B | Sample only from one class |
| 3 vs 8 ambiguous | Images where the model's 3/8 margin is < 0.25 |
| Low confidence | Images where max probability < 0.75 |
| Top-2 confusion | Images where the margin between top-2 classes < 0.20 |
| Custom pair | Your choice of two classes |

---

### Show True Label

**What it does:** Displays the ground-truth label from the MNIST test set alongside the prediction.

**Default:** On

When on, the history strip turns green for correct predictions and red for errors. Useful for seeing when drift causes misclassification.

---

## Reading the Drift Chart

The drift chart shows `delta_norm` (the accumulated weight change magnitude) over the last 120 predictions.

- A rising line means the model is drifting away from its neutral state.
- A flat line means drift is being balanced by decay.
- A falling line means return-to-normal is winning.
- Sudden drops to zero are manual resets.

If the line hits the `max_delta_norm` ceiling and flatlines there, the model is fully "saturated" — it is still predicting, but new updates are being rescaled to fit within the cap.

---

## A Note on What This Is Not

This is not training. The model does not improve on MNIST. The base weights never change.

What this is: a demonstration that a fixed inference function can exhibit short-term sequential biases — the output on step N is influenced by steps N-1, N-2, ... N-k — through a controlled, reversible, analytically understood mechanism in the final layer.

The analogy to human perception is informal. But the phenomenon being studied — that recent history biases current classification — is real and measurable here.
