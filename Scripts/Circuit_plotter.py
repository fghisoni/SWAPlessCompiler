import pennylane as qp
from pennylane.drawer import tape_mpl
import matplotlib.pyplot as plt

def draw_compiled_ops_pennylane(compiled_ops, debug, n_wires=None, decimals=3, show=True):
    """
    Draw a circuit from a compiled operation list using PennyLane's tape_mpl.
    No state vector allocation → works for many qubits (e.g., 52).

    Parameters
    ----------
    compiled_ops : list[list]
        Operation list in the format:
            ["RZ", [wire], angle]
            ["RX", [wire], angle]
            ["H", [wire], "N/A"]
            ["CNOT", [control, target], "N/A"]

    debug : dict
        Must contain a key 'placement_log' which is a list of dictionaries
        each with key 'spanning_line_index'. Used to insert qp.Barrier()
        between spanning lines.

    n_wires : int or None
        Number of physical wires. If None, inferred from compiled_ops.

    decimals : int
        Number of decimals shown for rotation angles.

    show : bool
        If True, call plt.show().

    Returns
    -------
    fig, ax
        Matplotlib figure and axis.
    """
    if n_wires is None:
        max_wire = -1
        for gate_type, wires, _ in compiled_ops:
            max_wire = max(max_wire, max(wires))
        n_wires = max_wire + 1

    # Record operations onto a tape (no simulation)
    with qp.tape.QuantumTape() as tape:
        current_spanning_line = 0
        idx = 0
        for gate_type, wires, angle in compiled_ops:
            try:
                if debug['placement_log'][idx]['spanning_line_index'] != current_spanning_line:
                    if n_wires < 50:
                        qp.Barrier()
                        current_spanning_line += 1
            except:
                pass

            gate_type = gate_type.upper()

            if gate_type == "RZ":
                if len(wires) != 1:
                    raise ValueError(f"RZ should act on one wire, got {wires}.")
                qp.RZ(angle, wires=wires[0])
                idx += 1

            elif gate_type == "RX":
                if len(wires) != 1:
                    raise ValueError(f"RX should act on one wire, got {wires}.")
                qp.RX(angle, wires=wires[0])
                idx += 1

            elif gate_type == "H":
                if len(wires) != 1:
                    raise ValueError(f"H should act on one wire, got {wires}.")
                qp.Hadamard(wires=wires[0])
                idx += 1

            elif gate_type == "CNOT":
                if len(wires) != 2:
                    raise ValueError(f"CNOT should act on two wires, got {wires}.")
                qp.CNOT(wires=wires)

            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")

    # Reverse wire order so qubit n-1 is at the top
    if debug['unitary'] != 'C^nX':
        wire_order = list(range(n_wires))
        wire_order.reverse()
        fig, ax = tape_mpl(tape, wire_order=wire_order, decimals=decimals)
    else: 
        fig, ax = tape_mpl(tape, wire_order=range(n_wires + 1), decimals=decimals)
    if show:
        plt.show()
    return fig, ax