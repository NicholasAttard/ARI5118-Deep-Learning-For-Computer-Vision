import streamlit as st
import pandas as pd
import numpy as np
import graphviz

# Page Setup
st.set_page_config(page_title="Backprop & Gradient Flow", layout="wide")
st.title("Backprop & Gradient Flow")

tab1, tab2 = st.tabs(["Backpropagation using Graph", "Softmax & Cross-Entropy"])

with tab1:
    st.header("The Chain Rule")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 'initial'

    # Input Sliders for a, b, c - from -10 to 10 with default values
    col1, col2, col3 = st.columns(3)
    a = col1.slider("Input (a)", -10.0, 10.0, 3.0)
    b = col2.slider("Input (b)", -10.0, 10.0, 1.0)
    c = col3.slider("Input (c)", -10.0, 10.0, -2.0)

    st.divider()

    # Forward Pass
    d = 2 * b
    e = a + d
    L = c * e

    # Backward Pass
    dL_dL = 1.0
    dL_de = c
    dL_dc = e
    de_da = 1.0
    de_dd = 1.0
    dd_db = 2.0

    # Calculated Gradients
    dL_da = dL_de * de_da
    dL_db = dL_de * de_dd * dd_db

    # Control Buttons to change mode between forward and backward passs
    col1, col2 = st.columns(2)
    if col1.button("Run Forward Pass"):
        st.session_state.step = 'forward'

    if col2.button("Run Backward Pass"):
        st.session_state.step = 'backward'

    # Graph Settup
    left_spacer, center_column, right_spacer = st.columns([1, 15, 1])
    dot = graphviz.Digraph()  
    dot.attr(rankdir='LR', size='10')
    dot.attr(nodesep='1.0', ranksep='2.0')
    
    # At the very top of your app
    if 'step' not in st.session_state:
        st.session_state.step = 'initial'

    is_back = st.session_state.step == 'backward'
    red = "#FF4B4B"
    green = "#29B045"
    blue = "#3B82F6"

    current_step = st.session_state.step

    if current_step == 'initial':
        node_color = blue
        is_back = False
    elif current_step == 'forward':
        node_color = green
        is_back = False
    else: # backward
        node_color = red
        is_back = True

    def get_label(name, val, grad=None):
        if current_step == 'backward' and grad is not None:
            return f"{name}\nVal: {val}\nGrad: {grad:.2f}"
        return f"{name}\nVal: {val}"
    
    # Define Nodes
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('A', get_label('a', a, dL_da), color=node_color, shape='circle')
        s.node('B', get_label('b', b, dL_db), color=node_color, shape='circle')
        s.node('C', get_label('c', c, dL_dc), color=node_color, shape='circle')
        
        # Makes the input circles in alphabetical order
        s.edge('A', 'B', style='invis')
        s.edge('B', 'C', style='invis')

        dot.node('D', get_label('d = 2b', d, dL_de * de_dd), color=node_color)
        dot.node('E', get_label('e = a + d', e, dL_de), color=node_color)
        dot.node('L', get_label('L = ce', L, dL_dL), color=node_color)

        # Define Edges
        dot.edge('A', 'E')
        dot.edge('B', 'D')
        dot.edge('D', 'E')
        dot.edge('E', 'L')
        dot.edge('C', 'L')

    # Center aligning the graph
    with center_column:
        st.graphviz_chart(dot, use_container_width=True)

    st.divider()

    # Explanation of Forward and Backward Passes calculations
    # Initial Partial Derivatives calculated
    if is_back:
        st.error("### Backward Pass")
        st.latex(rf"\frac{{\partial L}}{{\partial a}} = \frac{{\partial L}}{{\partial e}} \cdot \frac{{\partial e}}{{\partial a}} = {dL_de} \cdot {de_da} = {dL_da}")
        st.latex(rf"\frac{{\partial L}}{{\partial b}} = \frac{{\partial L}}{{\partial e}} \cdot \frac{{\partial e}}{{\partial d}} \cdot \frac{{\partial d}}{{\partial b}} = {dL_de} \cdot {de_dd} \cdot {dd_db} = {dL_db}")
        st.latex(rf"\frac{{\partial L}}{{\partial c}} = \frac{{\partial L}}{{\partial c}}  = {dL_dc}")
    
    elif st.session_state.step == 'forward':
        st.success("### Forward Pass")
        st.latex(rf"d = 2b = 2 \cdot {b} = {d}")
        st.latex(rf"e = a + d = {a} + {d} = {e}")
        st.latex(rf"L = ce = {c} \cdot {e} = {L}")

    else:
        st.info("### Partial Derivatives")
        # \quad to add tab, \qquad to add large tab
        st.latex(rf"L = ce : \quad \frac{{\partial L}}{{\partial c}} = e \quad,\quad \frac{{\partial L}}{{\partial e}} = c")
        st.latex(rf"L = a + d : \quad \frac{{\partial e}}{{\partial a}} = 1 \quad,\quad \frac{{\partial e}}{{\partial d}} = 1")
        st.latex(rf"d = 2b : \quad \frac{{\partial d}}{{\partial b}} = 2")


with tab2:
    st.header("Softmax & Cross-Entropy")

    st.info("### Softmax Function")
    st.latex(rf"\sigma(z_i) = \frac{{e^{{z_i}}}}{{\sum_{{j=1}}^K e^{{z_j}}}}")

    st.divider()

    st.subheader("Target ($y$)")
    # The internal tick boxes
    # We use a radio with a custom label to make it look like a selection
    target_class = st.radio("Select the correct class:", ["Class A", "Class B", "Class C"])

    st.divider()

    # Input Sliders for a, b, c - from -10 to 10 with default values
    col1, col2, col3 = st.columns(3)
    class1 = col1.slider("Output (class 1)", -5.0, 5.0, 2.0)
    class2 = col2.slider("Output (class 2)", -5.0, 5.0, 1.0)
    class3 = col3.slider("Output (class 3)", -5.0, 5.0, 0.1)

    def softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def cross_entropy(predictions, labels):
        return -np.sum(labels * np.log(predictions + 1e-15))

    logits = np.array([class1, class2, class3])
    true_label = np.array([1, 0, 0]) 

    st.divider()

    st.latex(rf"Logits: \quad z = [{class1}, {class2}, {class3}]")
    st.latex(rf"Softmax: \quad \sigma(z) = {softmax(logits)}")

    st.divider()

    st.info("### Cross-Entropy Loss")
    st.latex(rf"CrossEntropy(y, \hat{{y}}) = -\sum_{{i=1}}^K y_i \log(\hat{{y}}_i)")

    st.divider()

    st.latex(rf"Loss: \quad L = {cross_entropy(softmax(logits), true_label)}")  