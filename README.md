# Backpropagation — Interactive Walkthrough

An interactive Streamlit app that walks through backpropagation on a tiny 2-1-1 network, one step at a time. Built to make the chain-rule expansions feel more like a signal flowing backward through a graph.

![Python](https://img.shields.io/badge/python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/streamlit-1.30+-red) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

## What it shows

A minimal network — 2 inputs, 1 hidden neuron (tanh), 1 output neuron (sigmoid), squared-error loss:

```
x₁ ──┐
      ├──► G (tanh) ──► F (sigmoid) ──► e = ½(y − ŷ)²
x₂ ──┘
```

Five weights, five gradients. The app reveals the backward pass in 5 clickable steps:

1. **Forward pass** — push activations left → right, cache every intermediate.
2. **Error signal at F** — collapse the chain rule's shared prefix into a single scalar `δ_F`.
3. **F's weight gradients** — each one is just `δ_F × (input on the other side of the weight)`.
4. **Propagate to G** — push `δ` backward through the connecting weight: `δ_G = δ_F · w_gf1 · tanh'(z_G)`.
5. **G's weight gradients** — same universal rule: `δ × input`.

Edges turn red in the direction the error signal is flowing at each step. All inputs, weights, and the target are editable from the sidebar — every number recomputes live.

## The idea it's trying to make obvious

Textbook derivations expand `∂e/∂w` into 3-link or 5-link products of partials. Those products all share a common prefix. Backprop's trick is to **name that prefix `δ` and reuse it**, so every gradient collapses to:

> gradient = (δ on the output side of the weight) × (activation on the input side of the weight)

Once you see it on two neurons, the same rule scales to any feedforward network.

## Quick start

```bash
git clone https://github.com/<your-username>/backprop-walkthrough.git
cd backprop-walkthrough
pip install -r requirements.txt
streamlit run backprop_walkthrough.py
```

The app opens at `http://localhost:8501`.

### Graphviz system binary

`pip install graphviz` only installs the Python wrapper. You also need the Graphviz binary:

- **Windows**: `winget install graphviz` — then close and reopen your terminal. Verify with `dot -V`.
- **macOS**: `brew install graphviz`
- **Ubuntu/Debian**: `sudo apt install graphviz`

## Things to try

Once it's running, a few experiments that make the math click:

- **Set `y` equal to the current `ŷ`.** Every gradient goes to zero. This is why `(y − ŷ)` is called the error signal — no error, no learning.
- **Crank a weight up to push `z_F` toward ±10.** Watch `S'(z_F) = ŷ(1−ŷ)` collapse toward zero — `δ_F` shrinks even when the error is big. That's the **vanishing gradient problem** in its simplest possible form.
- **Set `w_gf1 = 0`.** All three of G's gradients die instantly. δ_G flows through `w_gf1`; no path means no learning signal reaches the hidden layer.
- **Set `x₁ = 0`.** `∂e/∂w_xg1` dies even though `w_xg1` is non-zero. Dead inputs produce dead gradients — that's why input normalization matters.

## Files

```
backprop_walkthrough.py    # the entire app, single file
requirements.txt           # streamlit, graphviz
README.md
LICENSE
.gitignore
```

## Why this exists

Made while working through the backprop textbooks show the chain-rule expansion correctly but the matrix notation hides the δ-reuse trick that makes backprop feel like an algorithm instead of an algebra exercise. This app surfaces that trick by reordering the presentation: compute δ once, then reuse.

## License

MIT — see [LICENSE](LICENSE).
