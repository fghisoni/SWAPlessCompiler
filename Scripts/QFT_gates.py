import numpy as np
from collections import OrderedDict

def generate_standard_qft_gates(n, include_final_swaps=False):
    """
    Generate the standard QFT circuit as a list of gate specifications.

    Each gate is represented as

        [gate_type, wires, angle]

    where:
        - gate_type is a string, e.g. "H", "CPHASE", "SWAP"
        - wires is a list:
            [a] for single-qubit gates,
            [control, target] for two-qubit controlled gates,
            [a, b] for SWAP gates
        - angle is a float if applicable, otherwise "N/A"

    The convention used is the standard QFT ordering

        for i = 0, ..., n-1:
            H on qubit i
            controlled phase gates between qubit j > i and qubit i
            with angle pi / 2^(j-i)

    The controlled phase is diagonal, so the distinction between control
    and target is mostly conventional. Here we use [j, i], meaning qubit j
    controls a phase on qubit i.

    Parameters
    ----------
    n : int
        Number of qubits.

    include_final_swaps : bool
        If True, include the final qubit-reversal SWAP network.
        If False, return the QFT without final swaps.

    Returns
    -------
    gates : list[list]
        List of gates in circuit implementation order.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    gates = []

    for i in range(n):
        # Hadamard on qubit i
        gates.append(["H", [i], "N/A"])

        # Controlled phase gates with qubits j > i
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            gates.append(["CPHASE", [j, i], angle])

    if include_final_swaps:
        for i in range(n // 2):
            gates.append(["SWAP", [i, n - 1 - i], "N/A"])

    return gates


def decompose_h_cphase_to_rx_rz(gates, keep_edge_hadamards=True):
    """
    Decompose a gate list containing Hadamards, controlled-phase gates,
    and SWAP gates.

    If keep_edge_hadamards is True, the first and last Hadamard gates
    (on qubit 0 and qubit n-1, where n is the maximum wire index+1)
    are left as H gates (not decomposed).

    Parameters
    ----------
    gates : list[list]
        List of gates in the format [gate_type, wires, angle].
    keep_edge_hadamards : bool
        If True, keep the first and last H gates as H (no decomposition).

    Returns
    -------
    decomposed_gates : list[list]
        List containing RX/RZ rotations, CNOTs, and possibly H gates.
    """
    # Determine number of qubits n from the maximum wire index
    max_wire = -1
    for gate in gates:
        gate_type, wires, _ = gate
        if gate_type.upper() in ("H", "CPHASE", "SWAP", "CNOT"):
            max_wire = max(max_wire, max(wires))
    n = max_wire + 1 if max_wire != -1 else 0

    decomposed_gates = []

    for gate in gates:
        gate_type, wires, angle = gate
        gate_type = gate_type.upper()

        if gate_type == "H":
            if len(wires) != 1:
                raise ValueError(f"Hadamard gate must act on one wire, got {wires}.")
            i = wires[0]

            # Keep the first (qubit 0) and last (qubit n-1) Hadamard if requested
            if keep_edge_hadamards and (i == 0 or i == n-1):
                decomposed_gates.append(["H", [i], "N/A"])
            else:
                decomposed_gates.append(["RZ", [i], np.pi / 2])
                decomposed_gates.append(["RX", [i], np.pi / 2])
                decomposed_gates.append(["RZ", [i], np.pi / 2])

        elif gate_type in ("CPHASE", "CP", "CONTROLLED_PHASE"):
            if len(wires) != 2:
                raise ValueError(
                    f"Controlled-phase gate must act on two wires, got {wires}."
                )
            if angle == "N/A":
                raise ValueError("Controlled-phase gate must have a numerical angle.")
            i, j = wires
            decomposed_gates.append(["RZ", [i], angle / 2])
            decomposed_gates.append(["RZ", sorted([i, j]), -angle / 2])
            decomposed_gates.append(["RZ", [j], angle / 2])

        elif gate_type == "SWAP":
            if len(wires) != 2:
                raise ValueError(f"SWAP gate must act on two wires, got {wires}.")
            i, j = wires
            decomposed_gates.append(["CNOT", [i, j], "N/A"])
            decomposed_gates.append(["CNOT", [j, i], "N/A"])
            decomposed_gates.append(["CNOT", [i, j], "N/A"])

        elif gate_type == "CNOT":
            if len(wires) != 2:
                raise ValueError(f"CNOT gate must act on two wires, got {wires}.")
            decomposed_gates.append(["CNOT", list(wires), "N/A"])

        else:
            raise ValueError(
                f"Unsupported gate type {gate_type}. "
                "Expected H, CPHASE, SWAP, or CNOT."
            )

    return decomposed_gates


def normalize_support(wires):
    """
    Convert a wire list/int into a canonical tuple.

    Examples
    --------
    [2, 0] -> (0, 2)
    [1]    -> (1,)
    1      -> (1,)
    """
    if isinstance(wires, int):
        return (wires,)
    return tuple(sorted(wires))


def pauli_rotations_commute(gate_a, gate_b):
    """
    Check whether two gates commute.

    Rules:
      - H does NOT commute with any gate (treated as a barrier).
      - RZ(S) commutes with RZ(T)
      - RX(S) commutes with RX(T)
      - RZ(S) commutes with RX(T) iff |S ∩ T| is even
    """
    type_a, wires_a, _ = gate_a
    type_b, wires_b, _ = gate_b

    type_a = type_a.upper()
    type_b = type_b.upper()

    # H gates are barriers: they do not commute with anything
    if type_a == "H" or type_b == "H":
        return False

    support_a = set(normalize_support(wires_a))
    support_b = set(normalize_support(wires_b))

    if type_a == type_b:
        return True

    if {type_a, type_b} == {"RZ", "RX"}:
        return len(support_a & support_b) % 2 == 0

    raise ValueError(f"Unsupported gate pair: {gate_a}, {gate_b}")


def flush_same_type_rotation_block(block, output, atol=0):
    """
    Combine same-type rotations in a commuting block.
    """
    if not block:
        return

    gate_type = block[0][0].upper()
    combined = OrderedDict()

    for _, wires, angle in block:
        support = normalize_support(wires)
        combined[support] = combined.get(support, 0.0) + angle

    for support, angle in combined.items():
        if abs(angle) > atol:
            output.append([gate_type, list(support), angle])


def combine_adjacent_same_type_rotations(operations, atol=0, combine_rx=True):
    """
    Combine adjacent blocks of same-type rotations.
    """
    output = []
    block = []
    block_type = None

    for gate in operations:
        gate_type, wires, angle = gate
        gate_type = gate_type.upper()

        is_combinable = gate_type == "RZ" or (combine_rx and gate_type == "RX")

        if not is_combinable:
            flush_same_type_rotation_block(block, output, atol=atol)
            block = []
            block_type = None
            output.append([gate_type, wires, angle])
            continue

        if block_type is None:
            block_type = gate_type
            block = [[gate_type, wires, angle]]

        elif gate_type == block_type:
            block.append([gate_type, wires, angle])

        else:
            flush_same_type_rotation_block(block, output, atol=atol)
            block_type = gate_type
            block = [[gate_type, wires, angle]]

    flush_same_type_rotation_block(block, output, atol=atol)

    return output


def move_rz_left_deterministic(operations):
    """
    Move each RZ gate as far left as possible across commuting gates,
    but preserve relative order among two-body RZ gates (support size 2).
    
    Single-qubit RZ gates acting on qubit 0 are never moved left.
    """
    output = []

    for gate in operations:
        gate = list(gate)
        gate[0] = gate[0].upper()

        if gate[0] != "RZ":
            output.append(gate)
            continue

        support = normalize_support(gate[1])
        is_single_qubit = len(support) == 1
        is_qubit0 = is_single_qubit and support[0] == 0

        # Do not move single-qubit RZ on qubit 0 to the left at all
        if is_qubit0:
            output.append(gate)
            continue

        insert_pos = len(output)
        is_two_body = len(support) == 2

        while insert_pos > 0:
            left_gate = output[insert_pos - 1]
            # Do not allow a two-body RZ to move past another two-body RZ
            if left_gate[0] == "RZ" and is_two_body and len(normalize_support(left_gate[1])) == 2:
                break
            if pauli_rotations_commute(gate, left_gate):
                insert_pos -= 1
            else:
                break

        output.insert(insert_pos, gate)

    return output

def first_rx_position_by_qubit(operations):
    """
    Return the position of the first single-qubit RX on each logical qubit.

    Example:
        RX [0] at position 3 gives out[(0,)] = 3.
    """
    out = {}

    for idx, gate in enumerate(operations):
        gate_type, wires, _ = gate
        gate_type = gate_type.upper()
        support = normalize_support(wires)

        if gate_type == "RX" and len(support) == 1 and support not in out:
            out[support] = idx

    return out

def move_single_rz_right_deterministic(
    operations,
    only_after_own_rx=True,
    pin_supports=None,
    always_move_right_qubits=None,
):
    """
    Move single-qubit RZ rotations as far right as possible across commuting gates.

    Parameters
    ----------
    operations : list[list]
        Gates in the format ["RZ", wires, angle] or ["RX", wires, angle].

    only_after_own_rx : bool
        If True, only move a single-qubit RZ to the right if it is already
        after the first RX acting on the same logical qubit. This preserves
        the useful front-loaded RZ rotations before their non-commuting RX.

    pin_supports : set[tuple] or None
        Optional set of supports that should not be moved right.
        Example: pin_supports={(0,)} keeps all RZ[0] gates fixed.

    always_move_right_qubits : set[int] or None
        Set of qubit indices for which the `only_after_own_rx` restriction is ignored.
        E.g., always_move_right_qubits={0} will move all single-qubit RZ[0] right
        regardless of whether they are after an RX[0].

    Returns
    -------
    ops : list[list]
        Reordered list.
    """
    ops = [list(op) for op in operations]
    pin_supports = set() if pin_supports is None else set(pin_supports)
    always_move = set() if always_move_right_qubits is None else set(always_move_right_qubits)

    rx_first_pos = first_rx_position_by_qubit(ops)

    # Scan right-to-left.
    for i in range(len(ops) - 1, -1, -1):
        gate = ops[i]
        gate_type, wires, _ = gate
        gate_type = gate_type.upper()
        support = normalize_support(wires)

        if gate_type != "RZ":
            continue

        if len(support) != 1:
            continue

        if support in pin_supports:
            continue

        qubit = support[0]

        # Decide whether to apply the "only after own RX" rule
        apply_after_rule = only_after_own_rx and (qubit not in always_move)

        if apply_after_rule:
            first_rx = rx_first_pos.get(support, None)
            if first_rx is None or i < first_rx:
                continue

        j = i
        while j + 1 < len(ops) and pauli_rotations_commute(ops[j], ops[j + 1]):
            ops[j], ops[j + 1] = ops[j + 1], ops[j]
            j += 1

    return ops

def compress_rx_rz_operations(
    operations,
    atol=0,
    combine_rx=True,
    pin_right_move_supports=None,
    always_move_right_qubits={0},   # new parameter
):
    """
    Compress RX/RZ operations in two stages.

    Parameters
    ----------
    operations : list[list]
        Input gates in format ["RZ", wires, angle] or ["RX", wires, angle].

    atol : float
        Remove rotations with absolute angle below this value.

    combine_rx : bool
        If True, combine adjacent RX rotations with identical support.

    pin_right_move_supports : set[tuple] or None
        Supports whose single-qubit RZ rotations should not be moved right.
        Example: {(0,)}.

    always_move_right_qubits : set[int] or None
        Qubits whose single-qubit RZ rotations should always be moved right,
        ignoring the `only_after_own_rx` rule. Example: {0}.

    Returns
    -------
    compressed : list[list]
        Equivalent compressed operation list.
    """
    ops = [list(op) for op in operations]

    # Stage 1: front-load commuting RZ rotations (preserving two‑body order)
    ops = move_rz_left_deterministic(ops)
    ops = combine_adjacent_same_type_rotations(
        ops,
        atol=atol,
        combine_rx=combine_rx,
    )

    # Stage 2: push remaining single-qubit RZ rotations to the right.
    ops = move_single_rz_right_deterministic(
        ops,
        only_after_own_rx=True,
        pin_supports=pin_right_move_supports,
        always_move_right_qubits=always_move_right_qubits,
    )
    ops = combine_adjacent_same_type_rotations(
        ops,
        atol=atol,
        combine_rx=combine_rx,
    )

    return ops