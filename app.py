# import streamlit as st
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
# from qiskit.visualization import plot_bloch_multivector, plot_state_qsphere
# import matplotlib.pyplot as plt
# import numpy as np

# st.title("Quantum Superposition and Entanglement Visualization")

# num_qubits = st.number_input("Number of Qubits", min_value=1, value=1, step=1)

# st.write("""
# This application demonstrates:
# - **Superposition:** For a single qubit, we use a Hadamard gate to create a state like (|0> + |1>)/√2.
# - **Entanglement:** For multiple qubits, we create an entangled state (e.g., a Bell state) using Hadamard and CNOT gates.

# **What you'll see:**
# - For 1 qubit: A Bloch sphere showing the qubit in a superposition state.
# - For multiple qubits: A Q-sphere showing all basis states, a probability distribution bar chart, and a calculation of the reduced density matrix purity to show entanglement.
# """)

# # Create the quantum circuit
# qc = QuantumCircuit(num_qubits)

# # Apply gates to create superposition and potentially entanglement
# if num_qubits == 1:
#     # Single qubit: just apply a Hadamard to get (|0> + |1>)/√2
#     qc.h(0)
# else:
#     # Multi-qubit:
#     # 1) Put the first qubit in superposition
#     qc.h(0)
#     # 2) Entangle with the second qubit if available
#     if num_qubits >= 2:
#         qc.cx(0, 1)
#     # 3) For any additional qubits, also put them into superposition
#     for q in range(2, num_qubits):
#         qc.h(q)

# # Obtain the statevector
# state = Statevector.from_instruction(qc)

# # Display the circuit
# st.write("**Quantum Circuit:**")
# fig_circuit = qc.draw(output='mpl')
# st.pyplot(fig_circuit)

# # Display the statevector
# st.write("**Statevector (amplitudes):**")
# st.text(state)

# # For visualization and analysis
# if num_qubits == 1:
#     # Single qubit visualization: Bloch sphere
#     st.write("**Bloch Sphere Representation:**")
#     fig_bloch = plot_bloch_multivector(state)
#     st.pyplot(fig_bloch)
#     st.write("""
#     The Bloch sphere shows the qubit in superposition if the state vector 
#     points away from the poles (which represent |0> and |1>).
#     """)
# else:
#     # Multiple qubits: Q-sphere and probabilities
#     st.write("**Q-Sphere Representation:**")
#     fig_qsphere = plot_state_qsphere(state)
#     st.pyplot(fig_qsphere)
    
#     # Probability distribution
#     st.write("**Probability Distribution of Computational Basis States:**")
#     probs = np.abs(state.data)**2
#     basis_states = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]
    
#     fig_prob, ax = plt.subplots()
#     ax.bar(basis_states, probs)
#     ax.set_xlabel("Basis State")
#     ax.set_ylabel("Probability")
#     ax.set_title("Probability Distribution")
#     st.pyplot(fig_prob)
    
#     # Check for entanglement: reduced density matrix of the first qubit
#     rho = DensityMatrix(state)
#     reduced_rho = partial_trace(rho, list(range(1, num_qubits)))
    
#     st.write("**Reduced Density Matrix of the First Qubit:**")
#     st.text(reduced_rho)
    
#     # Compute purity: Tr(rho^2)
     
#     0 # Use the underlying data (numpy array) for matrix multiplication
#     purity = np.trace(reduced_rho.data @ reduced_rho.data).real
#     st.write(f"Purity of the first qubit's reduced state: {purity:.4f}")

    
#     if purity < 1.0:
#         st.write("""
#         The reduced density matrix of the first qubit is mixed (purity < 1), indicating entanglement. 
#         This means the multi-qubit state cannot be factored into individual qubit states.
#         """)
#     else:
#         st.write("""
#         The reduced density matrix is pure (purity = 1). This would indicate no entanglement. 
#         However, given our chosen gates, it's unlikely unless the circuit was adjusted.
#         """)

import streamlit as st
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit.visualization import plot_bloch_multivector, plot_state_qsphere
import matplotlib.pyplot as plt
from qiskit.quantum_info import concurrence
import numpy as np

st.title("Quantum Superposition Visualization")

num_qubits = st.number_input("Number of Qubits", min_value=1, value=1, step=1)

st.write("""
This application demonstrates:
- **Superposition:** For a single qubit, we use a Hadamard gate to create a state like (|0> + |1>)/√2.
- **Entanglement:** For multiple qubits, we create an entangled state (e.g., a Bell state) using Hadamard and CNOT gates.

**Additional Features:**
- Apply custom gates to specific qubits.
- Visualize probability distributions with pie and bar charts.
- Analyze entanglement using purity and concurrence (for 2 qubits).
- Simulate measurements and view results.
- Generate and display Bell states.
- Perform classical-quantum hybrid execution.
""")

# Create the quantum circuit
qc = QuantumCircuit(num_qubits)

# Default gates (Hadamard and CNOT for entanglement)
if num_qubits == 1:
    qc.h(0)
else:
    qc.h(0)
    qc.cx(0, 1)
    for q in range(2, num_qubits):
        qc.h(q)

# User-defined gate application
st.write("### Add Custom Gates")
gate_options = ["X", "Y", "Z", "H", "CNOT"]
selected_gate = st.selectbox("Select a gate to apply:", gate_options)
target_qubit = st.number_input("Target Qubit (0-indexed):", min_value=0, max_value=num_qubits - 1, step=1)

if selected_gate == "CNOT":
    control_qubit = st.number_input("Control Qubit (0-indexed):", min_value=0, max_value=num_qubits - 1, step=1)
    if st.button("Apply Gate"):
        qc.cx(control_qubit, target_qubit)
else:
    if st.button("Apply Gate"):
        getattr(qc, selected_gate.lower())(target_qubit)

# Display the circuit
st.write("**Quantum Circuit:**")
fig_circuit = qc.draw(output='mpl')
st.pyplot(fig_circuit)

# Obtain the statevector
state = Statevector.from_instruction(qc)

# Display the statevector
st.write("**Statevector (amplitudes):**")
st.text(state)

# Visualization and analysis
if num_qubits == 1:
    st.write("**Bloch Sphere Representation:**")
    fig_bloch = plot_bloch_multivector(state)
    st.pyplot(fig_bloch)
else:
    st.write("**Q-Sphere Representation:**")
    fig_qsphere = plot_state_qsphere(state)
    st.pyplot(fig_qsphere)

    # Probability distribution: bar chart
    st.write("**Probability Distribution of Computational Basis States:**")
    probs = np.abs(state.data)**2
    basis_states = [bin(i)[2:].zfill(num_qubits) for i in range(2**num_qubits)]
    
    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(basis_states, probs)
    ax_bar.set_xlabel("Basis State")
    ax_bar.set_ylabel("Probability")
    ax_bar.set_title("Probability Distribution (Bar Chart)")
    st.pyplot(fig_bar)

    # Probability distribution: pie chart
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(probs, labels=basis_states, autopct='%1.1f%%', startangle=90)
    ax_pie.set_title("Probability Distribution (Pie Chart)")
    st.pyplot(fig_pie)

    # Reduced density matrix for entanglement check
    rho = DensityMatrix(state)
    reduced_rho = partial_trace(rho, list(range(1, num_qubits)))
    st.write("**Reduced Density Matrix of the First Qubit:**")
    st.text(reduced_rho)

    # Compute purity: Tr(rho^2)
    purity = np.trace(reduced_rho.data @ reduced_rho.data).real
    st.write(f"Purity of the first qubit's reduced state: {purity:.4f}")

    # Calculate concurrence for 2-qubit entanglement
    if num_qubits == 2:
        conc = concurrence(rho)
        st.write(f"Concurrence: {conc:.4f}")
        if conc > 0:
            st.write("The two-qubit system is entangled.")
        else:
            st.write("The two-qubit system is not entangled.")

    if purity < 1.0:
        st.write("""
        The reduced density matrix is mixed (purity < 1), indicating entanglement.
        """)
    else:
        st.write("""
        The reduced density matrix is pure (purity = 1), suggesting no entanglement.
        """)

# Dynamic Measurement Simulation
st.write("### Simulate Measurements")
shots = st.slider("Number of Shots:", min_value=100, max_value=10000, step=100, value=1000)
if st.button("Simulate Measurements"):
    backend = Aer.get_backend('qasm_simulator')
    qc.measure_all()
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Plot measurement results
    st.write("**Measurement Results:**")
    fig_measure, ax_measure = plt.subplots()
    ax_measure.bar(counts.keys(), counts.values())
    ax_measure.set_xlabel("Measurement Outcome")
    ax_measure.set_ylabel("Counts")
    ax_measure.set_title(f"Measurement Results ({shots} shots)")
    st.pyplot(fig_measure)

# Generate and Display Bell States
st.write("### Generate Bell States")
bell_state_options = ["|Φ+⟩ = (|00⟩ + |11⟩)/√2", "|Φ-⟩ = (|00⟩ - |11⟩)/√2", 
                      "|Ψ+⟩ = (|01⟩ + |10⟩)/√2", "|Ψ-⟩ = (|01⟩ - |10⟩)/√2"]
selected_bell_state = st.selectbox("Select a Bell State to Generate:", bell_state_options)

if st.button("Generate Bell State"):
    qc_bell = QuantumCircuit(2)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    if "Φ-" in selected_bell_state:
        qc_bell.z(0)
    elif "Ψ+" in selected_bell_state:
        qc_bell.x(0)
    elif "Ψ-" in selected_bell_state:
        qc_bell.z(0)
        qc_bell.x(0)

    st.write("**Generated Bell State Circuit:**")
    fig_bell = qc_bell.draw(output='mpl')
    st.pyplot(fig_bell)

# Classical-Quantum Hybrid Execution
st.write("### Classical-Quantum Hybrid Computation")
classical_input = st.number_input("Enter a Classical Input (integer):", value=2, step=1)

if st.button("Run Hybrid Computation"):
    qc.h(0)
    for i in range(classical_input):
        qc.rx(np.pi / 2, 0)
    st.write(f"**Circuit after {classical_input} classical iterations:**")
    fig_hybrid = qc.draw(output='mpl')
    st.pyplot(fig_hybrid)

    # Visualize the statevector after the hybrid computation
    state = Statevector.from_instruction(qc)
    st.write("**Statevector (amplitudes) after hybrid computation:**")
    st.text(state)

    # Optional: Simulate measurement
    backend = Aer.get_backend('qasm_simulator')
    qc.measure_all()
    job = backend.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    st.write("**Measurement Results:**")
    fig_measure, ax_measure = plt.subplots()
    ax_measure.bar(counts.keys(), counts.values())
    ax_measure.set_xlabel("Measurement Outcome")
    ax_measure.set_ylabel("Counts")
    ax_measure.set_title(f"Measurement Results (1000 shots)")
    st.pyplot(fig_measure)
