"""
Interactive backprop walkthrough for the 2-1-1 network.

Run:
    pip install streamlit
    streamlit run backprop_walkthrough.py
"""

import math
import streamlit as st
import graphviz

st.set_page_config(page_title="Backprop walkthrough", layout="wide")
st.title("Backpropagation — step by step")
st.caption("2 inputs → 1 hidden neuron (tanh) → 1 output neuron (sigmoid) → squared error")


# ---------- sidebar: inputs, weights, target ----------
with st.sidebar:
    st.header("Inputs")
    x1 = st.number_input("x₁", value=0.5, step=0.1, format="%.2f")
    x2 = st.number_input("x₂", value=-0.3, step=0.1, format="%.2f")
    y  = st.number_input("y (target)", value=1.0, step=0.1, format="%.2f")

    st.header("Weights at G (hidden)")
    w_xg0 = st.number_input("w_xg0 (bias)", value=0.1, step=0.1, format="%.2f")
    w_xg1 = st.number_input("w_xg1", value=0.8, step=0.1, format="%.2f")
    w_xg2 = st.number_input("w_xg2", value=-0.5, step=0.1, format="%.2f")

    st.header("Weights at F (output)")
    w_gf0 = st.number_input("w_gf0 (bias)", value=0.2, step=0.1, format="%.2f")
    w_gf1 = st.number_input("w_gf1", value=0.9, step=0.1, format="%.2f")


# ---------- the actual math ----------
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# forward pass
z_G   = w_xg0 + w_xg1 * x1 + w_xg2 * x2
y_G   = math.tanh(z_G)
z_F   = w_gf0 + w_gf1 * y_G
y_hat = sigmoid(z_F)
err   = y - y_hat
e     = 0.5 * err ** 2

# local activation derivatives (cached from forward pass)
S_prime    = y_hat * (1 - y_hat)     # sigmoid'(z_F) expressed via ŷ
tanh_prime = 1 - y_G ** 2            # tanh'(z_G) expressed via y_G

# the two deltas — the whole point of backprop
delta_F = -err * S_prime                         # ∂e/∂z_F
delta_G = delta_F * w_gf1 * tanh_prime           # ∂e/∂z_G

# gradients: δ_on_output_side × activation_on_input_side
g_gf0 = delta_F * 1
g_gf1 = delta_F * y_G
g_xg0 = delta_G * 1
g_xg1 = delta_G * x1
g_xg2 = delta_G * x2


# ---------- step controller ----------
step = st.slider(
    "Step",
    min_value=0, max_value=5, value=0,
    help="0: reset · 1: forward · 2: error at F · 3: F's grads · 4: push to G · 5: G's grads",
)


# ---------- network diagram (graphviz) ----------
def build_graph(step):
    g = graphviz.Digraph()
    g.attr(rankdir="LR", bgcolor="transparent", nodesep="0.4", ranksep="0.6")
    g.attr("node", fontname="Helvetica", fontsize="11")
    g.attr("edge", fontname="Helvetica", fontsize="9")

    fwd_color = "#185FA5" if step >= 1 else "#888780"
    back_to_F = "#E24B4A" if step >= 2 else fwd_color
    back_to_G = "#E24B4A" if step >= 4 else fwd_color
    back_to_x = "#E24B4A" if step >= 5 else fwd_color

    g.node("x1", f"x₁\n{x1:.2f}", shape="circle", style="filled", fillcolor="#F1EFE8")
    g.node("x2", f"x₂\n{x2:.2f}", shape="circle", style="filled", fillcolor="#F1EFE8")
    g.node("b1", "1", shape="circle", style="filled", fillcolor="#F1EFE8", width="0.3")
    g.node("b2", "1", shape="circle", style="filled", fillcolor="#F1EFE8", width="0.3")

    G_label = f"G (tanh)\nz_G={z_G:.3f}\ny_G={y_G:.3f}" if step >= 1 else "G (tanh)"
    F_label = f"F (sigmoid)\nz_F={z_F:.3f}\nŷ={y_hat:.3f}" if step >= 1 else "F (sigmoid)"
    E_label = f"e\n{e:.4f}" if step >= 1 else "e"

    g.node("G", G_label, shape="circle", style="filled", fillcolor="#EEEDFE")
    g.node("F", F_label, shape="circle", style="filled", fillcolor="#E1F5EE")
    g.node("E", E_label, shape="circle", style="filled", fillcolor="#FAECE7")

    g.edge("x1", "G", label=f"w_xg1={w_xg1:.2f}", color=back_to_x, penwidth="1.5")
    g.edge("x2", "G", label=f"w_xg2={w_xg2:.2f}", color=back_to_x, penwidth="1.5")
    g.edge("b1", "G", label=f"w_xg0={w_xg0:.2f}", color=back_to_x, style="dashed")
    g.edge("G",  "F", label=f"w_gf1={w_gf1:.2f}", color=back_to_G, penwidth="1.5")
    g.edge("b2", "F", label=f"w_gf0={w_gf0:.2f}", color=back_to_G, style="dashed")
    g.edge("F",  "E", color=back_to_F, penwidth="1.5")
    return g


col_net, col_info = st.columns([3, 2])

with col_net:
    st.graphviz_chart(build_graph(step), use_container_width=True)

with col_info:
    st.metric("ŷ (predicted)", f"{y_hat:.4f}" if step >= 1 else "—")
    st.metric("y − ŷ",          f"{err:.4f}"   if step >= 1 else "—")
    st.metric("e = ½(y−ŷ)²",    f"{e:.4f}"     if step >= 1 else "—")
    if step >= 2:
        st.metric("δ_F = ∂e/∂z_F", f"{delta_F:.4f}")
    if step >= 4:
        st.metric("δ_G = ∂e/∂z_G", f"{delta_G:.4f}")


# ---------- per-step explanation ----------
explanations = {
    0: ("Step 0 — start",
        "Move the slider to step 1 to push the inputs through the network. "
        "Every number computed in the forward pass gets **cached** — backprop "
        "reuses them instead of recomputing."),
    1: ("Step 1 — forward pass",
        f"Signals flow left → right.  \n"
        f"z_G = {w_xg0:.2f} + {w_xg1:.2f}·{x1:.2f} + {w_xg2:.2f}·{x2:.2f} = **{z_G:.4f}**  \n"
        f"y_G = tanh(z_G) = **{y_G:.4f}**  \n"
        f"z_F = {w_gf0:.2f} + {w_gf1:.2f}·{y_G:.4f} = **{z_F:.4f}**  \n"
        f"ŷ = S(z_F) = **{y_hat:.4f}**  \n"
        f"e = ½(y − ŷ)² = **{e:.4f}**"),
    2: ("Step 2 — error signal arrives at F",
        f"The error travels backward not as a long chain but as a **single scalar** "
        f"δ_F = ∂e/∂z_F. This is the shared prefix of every gradient downstream.  \n\n"
        f"δ_F = −(y − ŷ)·S'(z_F) = −({err:.4f}) × {S_prime:.4f} = **{delta_F:.4f}**  \n\n"
        f"Note S'(z_F) = ŷ(1−ŷ) — the derivative of sigmoid expressed in terms of its "
        f"output, so no extra computation needed."),
    3: ("Step 3 — F's weight gradients",
        f"Both of F's gradients follow the same rule: **δ on the output side × "
        f"activation on the input side**.  \n\n"
        f"∂e/∂w_gf0 = δ_F · 1    = **{g_gf0:.4f}**  (bias input is 1)  \n"
        f"∂e/∂w_gf1 = δ_F · y_G  = **{g_gf1:.4f}**  \n\n"
        f"No matrix of partials. No summing over paths. Just take the δ that arrived at F and multiply "
        f"one multiply."),
    4: ("Step 4 — push δ backward through w_gf1",
        f"To propagate to G: take δ_F, multiply by the **weight connecting F and G**, "
        f"then by G's local activation derivative.  \n\n"
        f"δ_G = δ_F · w_gf1 · tanh'(z_G)  \n"
        f"      = {delta_F:.4f} × {w_gf1:.2f} × {tanh_prime:.4f}  \n"
        f"      = **{delta_G:.4f}**  \n\n"
        f"This is the **propagation** step. tanh'(z_G) = 1 − y_G² — again, derivative "
        f"written in terms of forward-pass outputs."),
    5: ("Step 5 — G's weight gradients",
        f"Same rule as F: δ_G × input on that weight's input side.  \n\n"
        f"∂e/∂w_xg0 = δ_G · 1   = **{g_xg0:.4f}**  (bias)  \n"
        f"∂e/∂w_xg1 = δ_G · x₁  = **{g_xg1:.4f}**  \n"
        f"∂e/∂w_xg2 = δ_G · x₂  = **{g_xg2:.4f}**  \n\n"
        f"The 5-link chain from slide 3 disappears — δ_G already absorbed everything "
        f"to the right of G."),
}

title, body = explanations[step]
st.subheader(title)
st.markdown(body)


# ---------- final gradient summary ----------
if step >= 5:
    st.divider()
    st.subheader("All five gradients — the final answer")
    st.markdown(
        "| Weight | Gradient formula | Value |\n"
        "|---|---|---|\n"
        f"| w_gf0 | δ_F · 1     | **{g_gf0:+.4f}** |\n"
        f"| w_gf1 | δ_F · y_G   | **{g_gf1:+.4f}** |\n"
        f"| w_xg0 | δ_G · 1     | **{g_xg0:+.4f}** |\n"
        f"| w_xg1 | δ_G · x₁    | **{g_xg1:+.4f}** |\n"
        f"| w_xg2 | δ_G · x₂    | **{g_xg2:+.4f}** |\n"
    )
    st.info(
        "**The universal rule:** for any weight in any feedforward network, "
        "gradient = (δ on output side) × (activation on input side). "
        "Both numbers are already sitting in memory from the two passes."
    )