import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sympy as sp

# Page Setup
st.set_page_config(page_title="Backprop & Gradient Flow", layout="wide")
st.title("Backprop & Gradient Flow")

tab1, tab2, tab3 = st.tabs(["Backpropagation using Graph", "Softmax & Cross-Entropy", "Neural Network Weight Update and Loss Calculation"])

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

    is_back = st.session_state.step == 'backward'
    red = "#FF4B4B"
    green = "#29B045"
    blue = "#3B82F6"

     # Control Buttons to change mode between forward and backward passs
    col1, col2 = st.columns(2)
    if col1.button("Run Forward Pass"):
        st.session_state.step = 'forward'

    if col2.button("Run Backward Pass"):
        st.session_state.step = 'backward'
        is_back = st.session_state.step == 'backward'

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

    
    st.info("### Partial Derivatives")
    # \quad to add tab, \qquad to add large tab
    st.latex(rf"L = ce : \quad \frac{{\partial L}}{{\partial c}} = e \quad,\quad \frac{{\partial L}}{{\partial e}} = c")
    st.latex(rf"L = a + d : \quad \frac{{\partial e}}{{\partial a}} = 1 \quad,\quad \frac{{\partial e}}{{\partial d}} = 1")
    st.latex(rf"d = 2b : \quad \frac{{\partial d}}{{\partial b}} = 2")

    st.divider()

    # Graph Settup
    left_spacer, center_column, right_spacer = st.columns([1, 15, 1])
    dot = graphviz.Digraph()  
    dot.attr(rankdir='LR', size='10')
    dot.attr(nodesep='1.0', ranksep='2.0')
    
    if 'step' not in st.session_state:
        st.session_state.step = 'initial'

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
        
        if current_step == 'forward':
            return f"{name}\nVal: {val}"
        
        return f"{name}"
    
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

    st.info("### Gradient Descent to Update Weights")

    #Session State
    if 'x_curr' not in st.session_state:
        st.session_state.x_curr = 0.0
    if 'iteration' not in st.session_state:
        st.session_state.iteration = 0
    if 'history' not in st.session_state:
        st.session_state.history = []

   
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuration")
        
        #Function Input
        raw_func = st.text_input("Function", value="x**2 + 4")
        
        #Validate function
        try:
            x_sym = sp.symbols('x')
            # This converts the string into a math expression
            f_sym = sp.simplify(raw_func) 
            df_sym = sp.diff(f_sym, x_sym)
            
            # Convert SymPy expressions to fast numerical functions
            f_num = sp.lambdify(x_sym, f_sym, "numpy")
            df_num = sp.lambdify(x_sym, df_sym, "numpy")
            
        except Exception as e:
            st.error("Invalid Function! Use Python syntax (e.g., x**2 for x²)")
            st.stop()

        start_point = st.number_input("Starting Point", value=5.0)
        learning_rate = st.number_input("Learning Rate", value=0.1, step=0.01, format="%.2f")

        if st.button("Set Up", use_container_width=True):
            st.session_state.x_curr = float(start_point)
            st.session_state.iteration = 0
            st.session_state.history = [float(start_point)]

        if st.button("Next Iteration", use_container_width=True):
            grad = df_num(st.session_state.x_curr)
            st.session_state.x_curr -= learning_rate * grad
            st.session_state.iteration += 1
            st.session_state.history.append(st.session_state.x_curr)

    with col2:
        st.subheader(f"Iteration: {st.session_state.iteration}")
        
        # Dynamic axis scaling based on start point
        limit = max(abs(start_point) * 1.5, 10)
        x_range = np.linspace(-limit, limit, 500)
        
        try:
            y_range = f_num(x_range)
            
            fig, ax = plt.subplots()
            ax.plot(x_range, y_range, color="steelblue", lw=2, label=f"f(x) = {raw_func}")
            
            if st.session_state.history:
                h_x = np.array(st.session_state.history)
                h_y = f_num(h_x)
                
                #ax.plot(h_x, h_y, color="orange", linestyle="--", alpha=0.8)
                ax.scatter(h_x, h_y, color="red", s=15)
                
                curr_x = st.session_state.x_curr
                curr_y = f_num(curr_x)
                slope = df_num(curr_x)
                
                ax.scatter([curr_x], [curr_y], color="green", s=10, zorder=5)
                
                tan_x = np.linspace(curr_x - 5, curr_x + 5, 10)
                ax.plot(tan_x, slope*(tan_x - curr_x) + curr_y, color="coral", lw=2)

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.axhline(0, color='black', lw=0.8)
            ax.axvline(0, color='black', lw=0.8)
            for spine in ax.spines.values(): spine.set_visible(False)
            
            st.pyplot(fig)
            st.metric("Current Value", f"{st.session_state.x_curr:.4f}")
            
        except Exception as e:
            st.warning("Could not plot function. Check your math syntax!")


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




with tab3:

    # --- Page Setup ---
    st.set_page_config(page_title="NN Visualizer", layout="wide")

    # --- Math Functions ---
    def relu(x): return np.maximum(0, x)
    def relu_deriv(x): return (x > 0).astype(float)

    def softmax(z):
        shift_z = z - np.max(z)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=0)


    def initialize_nn():
        return {
            # Weights: He Initialization sqrt(2/fan_in)
            'W1': np.random.randn(3, 2) * np.sqrt(2 / 2),
            'W2': np.random.randn(2, 3) * np.sqrt(2 / 3),
            
            # Biases: Explicitly zero
            'b1': np.zeros((3, 1)),
            'b2': np.zeros((2, 1)),
            
            # Momentum Velocities: Must be zero
            'vW1': np.zeros((3, 2)),
            'vb1': np.zeros((3, 1)),
            'vW2': np.zeros((2, 3)),
            'vb2': np.zeros((2, 1)),
            'a1': np.zeros((3, 1)),
            'a2': np.zeros((2, 1)),
            'step': "Ready",
            'loss_history': []
        }

    if 'nn' not in st.session_state:
        st.session_state.nn = initialize_nn()

    nn = st.session_state.nn

    st.title("Interactive Neural Network Simulator")
    st.write("Adjust inputs and parameters below, then walk through the training steps.")

    input_col, param_col = st.columns(2)

    with input_col:
        st.subheader("Data Inputs")
        x1 = st.slider("Input 1 (x1)", -2.0, 2.0, 1.0)
        x2 = st.slider("Input 2 (x2)", -2.0, 2.0, 0.5)
        x_in = np.array([[x1], [x2]])
        
        target_idx = st.select_slider("Target Class (Which node should be 1.0?)", options=[0, 1])
        y_target = np.zeros((2, 1))
        y_target[target_idx] = 1.0

    with param_col:
        st.subheader("Hyperparameters")
        lr = st.slider("Learning Rate", 0.01, 1.0, 0.2)
        
        if st.button("Reset Network & History", use_container_width=True):
            st.session_state.nn = initialize_nn()
            st.rerun()

    st.divider()
    step_col1, step_col2, step_col3 = st.columns(3)

    if step_col1.button("1. Forward Pass", use_container_width=True):
        nn['z1'] = np.dot(nn['W1'], x_in) + nn['b1']
        nn['a1'] = np.maximum(0, nn['z1'])
        
        nn['z2'] = np.dot(nn['W2'], nn['a1']) + nn['b2']
        shift_z2 = nn['z2'] - np.max(nn['z2'])
        nn['a2'] = np.exp(shift_z2) / np.sum(np.exp(shift_z2), axis=0)
        
        nn['step'] = "Forward"

    if step_col2.button("2. Backward Pass", use_container_width=True):
        if 'a2' in nn:
            dz2 = nn['a2'] - y_target
            nn['dW2'] = np.dot(dz2, nn['a1'].T)
            dz1 = np.dot(nn['W2'].T, dz2) * relu_deriv(nn['z1'])
            nn['dW1'] = np.dot(dz1, x_in.T)
            nn['step'] = "Backward"
        else:
            st.warning("Please run Forward Pass first.")

    if step_col3.button("3. Update Weights", use_container_width=True):
        if 'dW1' in nn:
            beta = 0.9
            
            nn['vW1'] = beta * nn['vW1'] + (1 - beta) * nn['dW1']
            nn['W1'] -= lr * nn['vW1']
            
            nn['vW2'] = beta * nn['vW2'] + (1 - beta) * nn['dW2']
            nn['W2'] -= lr * nn['vW2']
            
            loss = -np.log(nn['a2'][target_idx, 0] + 1e-9)
            nn['loss_history'].append(loss)
            nn['step'] = "Updated with Momentum"

    if st.button("Run Full Iteration", use_container_width=True):
        #Forward Pass
        nn['z1'] = np.dot(nn['W1'], x_in) + nn['b1']
        nn['a1'] = np.maximum(0, nn['z1']) 
        nn['z2'] = np.dot(nn['W2'], nn['a1']) + nn['b2']
        shift_z2 = nn['z2'] - np.max(nn['z2'])
        nn['a2'] = np.exp(shift_z2) / np.sum(np.exp(shift_z2), axis=0)
        
        #Backward Pass
        dz2 = nn['a2'] - y_target
        nn['dW2'] = np.dot(dz2, nn['a1'].T)
        nn['db2'] = dz2
        dz1 = np.dot(nn['W2'].T, dz2) * (nn['z1'] > 0).astype(float)
        nn['dW1'] = np.dot(dz1, x_in.T)
        nn['db1'] = dz1
        
        #Update Weights
        beta = 0.9
        nn['vW1'] = beta * nn['vW1'] + (1 - beta) * nn['dW1']
        nn['vb1'] = beta * nn['vb1'] + (1 - beta) * nn['db1']
        nn['W1'] -= lr * nn['vW1']
        nn['b1'] -= lr * nn['vb1']
        
        nn['vW2'] = beta * nn['vW2'] + (1 - beta) * nn['dW2']
        nn['vb2'] = beta * nn['vb2'] + (1 - beta) * nn['db2']
        nn['W2'] -= lr * nn['vW2']
        nn['b2'] -= lr * nn['vb2']
        
        #Record Loss
        loss = -np.log(nn['a2'][target_idx, 0] + 1e-9)
        nn['loss_history'].append(loss)
        nn['step'] = "Full Iteration Complete"

    def draw_network():
        # Increase DPI and figsize for better text clarity
        fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
        ax.set_aspect('equal')
        ax.axis('off')

        # Spread: In at -5, Hidden at 0, Out at 5
        cols = {'in': -5, 'hid': 0, 'out': 5}
        y_coords = {'in': [1.5, -1.5], 'hid': [2.5, 0, -2.5], 'out': [1.5, -1.5]}
        
        #Weights & Connections
        for i, yi in enumerate(y_coords['in']):
            for j, yj in enumerate(y_coords['hid']):
                ax.plot([cols['in'], cols['hid']], [yi, yj], color='gray', lw=1, alpha=0.3, zorder=1)
                t, stagger = 0.3, (0.2 if i == 0 else -0.2)
                ax.text(cols['in'] + (cols['hid']-cols['in'])*t, yi + (yj-yi)*t + stagger, 
                        f"{nn['W1'][j,i]:.2f}", color='blue', fontsize=9, ha='center',
                        fontweight='bold', zorder=5, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        for i, yi in enumerate(y_coords['hid']):
            for j, yj in enumerate(y_coords['out']):
                ax.plot([cols['hid'], cols['out']], [yi, yj], color='gray', lw=1, alpha=0.3, zorder=1)
                t, stagger = 0.7, ([0.25, 0, -0.25][i])
                ax.text(cols['hid'] + (cols['out']-cols['hid'])*t, yi + (yj-yi)*t + stagger, 
                        f"{nn['W2'][j,i]:.2f}", color='blue', fontsize=9, ha='center',
                        fontweight='bold', zorder=5, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))

        #Input Nodes
        for i, y in enumerate(y_coords['in']):
            ax.add_patch(patches.Circle((cols['in'], y), 0.75, fc='white', ec='black', lw=2, zorder=10))
            ax.text(cols['in'], y, f"{x_in[i,0]:.2f}", 
                    ha='center', va='center', color='black', fontweight='bold', fontsize=11, zorder=20)

        #Hidden Nodes
        for i, y in enumerate(y_coords['hid']):
            val = nn.get('a1', np.zeros((3,1)))[i,0]
            node_color = '#6a0dad' if val > 0 else '#4169e1' # Purple if active, Blue if inactive
            
            ax.add_patch(patches.Circle((cols['hid'], y), 0.75, fc=node_color, ec='black', lw=2, zorder=10))
            ax.text(cols['hid'], y, f"{val:.2f}", 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11, zorder=20)

        # Output Nodes
        for i, y in enumerate(y_coords['out']):
            prob_val = nn.get('a2', np.zeros((2,1)))[i, 0]
            edge_c = 'red' if i == target_idx else 'black'
            
            ax.add_patch(patches.Circle((cols['out'], y), 0.75, fc='white', ec=edge_c, lw=3, zorder=10))
            ax.text(cols['out'], y, f"P({i})\n{prob_val:.4f}", 
                    ha='center', va='center', color='black', fontweight='bold', fontsize=11, zorder=20)

        st.pyplot(fig)
   
    draw_network()

    st.divider()
    #Metrics and History
    if nn['loss_history']:
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Current Loss", f"{nn['loss_history'][-1]:.4f}")
        with m2:
            st.line_chart(nn['loss_history'])