# SWAPlessCompiler

This repository contains a Python implementation of swap-less compilation techniques based on temporal parity computing and the extended LHZ architecture. The code was developed to study how quantum algorithms can be compiled on limited-connectivity hardware without explicitly inserting SWAP gates.

The repository currently includes examples for:

- the Quantum Fourier Transform (QFT);
- the Quantum Approximate Optimization Algorithm (QAOA), with separate compilation of the problem unitary \(U_P\) and mixer unitary \(U_X\);
- the Toffoli gate, implemented through a phase-gadget decomposition of the \(CCZ\) gate and compiled using LHZ spanning lines and an ancilla-assisted implementation of the remaining three-body \(Z\) rotation.

The implementation follows the idea of tracking logical parity labels through a sequence of spanning lines in the extended LHZ layout. Logical operations are applied when the required parity label or logical line becomes local on the current physical qubits.

---

## Repository structure

```text
SWAPlessCompiler/
├── Notebooks/
│   ├── QAOA.ipynb
│   ├── QFT.ipynb
│   └── Toffoli.ipynb
├── Scripts/
│   ├── __init__.py
│   ├── Circuit_plotter.py
│   ├── Compiler.py
│   ├── LHZ.py
│   ├── QAOA_compiler.py
│   ├── QAOA_gates.py
│   ├── QAOA_spanning_lines.py
│   ├── QFT_compiler.py
│   ├── QFT_gates.py
│   ├── QFT_spanning_lines.py
│   ├── Spanning_lines.py
│   ├── Toffoli_compiler.py
│   ├── Toffoli_gates.py
│   └── Toffoli_spanning_lines.py
├── requirements.txt
└── README.md
└── SWAPlesscompiler.pdf