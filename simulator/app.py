import streamlit as st
import pandas as pd
import numpy as np
import graphviz

# Page Setup
st.set_page_config(page_title="Backprop & Gradient Flow", layout="wide")
st.title("Backprop & Gradient Flow")

tab1, tab2 = st.tabs(["Computational Graph", "Convolution & Softmax"])

with tab1:
    st.header("The Chain Rule")

    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 'nothing'

    # Input Sliders for a, b, c - from -10 to 10 with default values
    col1, col2, col3 = st.columns(3)
    a = col1.slider("Input (a)", -10.0, 10.0, 3.0)
    b = col2.slider("Input (b)", -10.0, 10.0, 1.0)
    c = col3.slider("Input (c)", -10.0, 10.0, -2.0)

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
    
    is_back = st.session_state.step == 'backward'
    red = "#FF4B4B"
    green = "#29B045"

    # Node Styling Functions
    def get_label(name, val, grad=None):
        if is_back and grad is not None:
            return f"{name}\nVal: {val}\nGrad: {grad}"
        return f"{name}\nVal: {val}"
    
    # Define Nodes
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('A', get_label('a', a, dL_da), color=red if is_back else green, shape='circle')
        s.node('B', get_label('b', b, dL_db), color=red if is_back else green, shape='circle')
        s.node('C', get_label('c', c, dL_dc), color=red if is_back else green, shape='circle')
        
        # Makes the input circles in alphabetical order
        s.edge('A', 'B', style='invis')
        s.edge('B', 'C', style='invis')

        dot.node('D', get_label('d = 2b', d, dL_de * de_dd), color=red if is_back else green)
        dot.node('E', get_label('e = a + d', e, dL_de), color=red if is_back else green)
        dot.node('L', get_label('L = ce', L, dL_dL), color=red if is_back else green)

        # Define Edges
        dot.edge('A', 'E')
        dot.edge('B', 'D')
        dot.edge('D', 'E')
        dot.edge('E', 'L')
        dot.edge('C', 'L')

    # Center aligning the graph
    with center_column:
        st.graphviz_chart(dot, use_container_width=True)

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