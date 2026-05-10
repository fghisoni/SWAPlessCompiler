from Scripts.LHZ import normalize_label


def rz_physical_qubit(logical_wires, spanning_line):
    """
    Find the physical qubit on which a logical/parity RZ rotation is local.

    For an RZ operation on logical support S, the operation is local if the
    current spanning line contains the parity label S.

    Parameters
    ----------
    logical_wires : list, tuple, set, or int
        Logical support of the RZ operation. Examples: [0], [0, 2].

    spanning_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    int or None
        Physical qubit carrying the requested parity label, or None if the
        operation is not local in the current spanning line.
    """
    target_label = normalize_label(logical_wires)

    matches = [
        q for q, label in enumerate(spanning_line)
        if normalize_label(label) == target_label
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def rx_physical_qubit(logical_wires, spanning_line):
    """
    Find the physical qubit on which a logical RX rotation is local.

    A logical RX_j is local if the logical index j appears in exactly one
    parity label of the current spanning line.

    Parameters
    ----------
    logical_wires : list, tuple, set, or int
        Logical qubit acted on by the RX operation. Must represent a singleton.

    spanning_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    int or None
        Physical qubit on which RX_j is local, or None if RX_j is not local.
    """
    logical_label = normalize_label(logical_wires)

    if len(logical_label) != 1:
        raise ValueError(
            f"Only single-logical-qubit RX operations are supported. "
            f"Got RX on {logical_wires}."
        )

    j = logical_label[0]

    matches = [
        q for q, label in enumerate(spanning_line)
        if j in normalize_label(label)
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def h_physical_qubit(logical_wires, spanning_line):
    """
    Find the physical qubit on which a logical H operation is local.

    Since H is not simply a Z-parity or X-line operation, this function only
    treats H_j as directly local when the current spanning line contains the
    singleton label (j,).

    Parameters
    ----------
    logical_wires : list, tuple, set, or int
        Logical qubit acted on by the H operation. Must represent a singleton.

    spanning_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    int or None
        Physical qubit carrying singleton label (j,), or None if unavailable.
    """
    logical_label = normalize_label(logical_wires)

    if len(logical_label) != 1:
        raise ValueError(
            f"Only single-logical-qubit H operations are supported. "
            f"Got H on {logical_wires}."
        )

    matches = [
        q for q, label in enumerate(spanning_line)
        if normalize_label(label) == logical_label
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def operation_physical_qubit(operation, spanning_line):
    """
    Check whether a logical operation is local in the current spanning line.

    Supported operations:
        ["RZ", wires, angle]
        ["RX", wires, angle]
        ["H", wires, "N/A"]

    Parameters
    ----------
    operation : list
        Logical operation in the format [gate_type, wires, angle].

    spanning_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    int or None
        Physical qubit on which the operation should be applied, or None if
        the operation is not local.
    """
    gate_type, wires, _ = operation
    gate_type = gate_type.upper()

    if gate_type == "RZ":
        return rz_physical_qubit(wires, spanning_line)

    if gate_type == "RX":
        return rx_physical_qubit(wires, spanning_line)

    if gate_type == "H":
        return h_physical_qubit(wires, spanning_line)

    raise ValueError(
        f"Unsupported operation type {gate_type}. Expected H, RX, or RZ."
    )


def logical_operations_commute(op_a, op_b):
    """
    Check whether two logical operations commute for scheduling purposes.

    Supported operations:
        ["RZ", wires, angle]
        ["RX", wires, angle]
        ["H", wires, "N/A"]

    Rules
    -----
    - RZ(S) commutes with RZ(T).
    - RX(S) commutes with RX(T).
    - RZ(S) commutes with RX(T) iff |S intersection T| is even.
    - H is treated as non-commuting with any other operation for scheduling.
    """
    type_a, wires_a, _ = op_a
    type_b, wires_b, _ = op_b

    type_a = type_a.upper()
    type_b = type_b.upper()

    if type_a == "H" or type_b == "H":
        return False

    support_a = set(normalize_label(wires_a))
    support_b = set(normalize_label(wires_b))

    if type_a == type_b:
        return True

    if {type_a, type_b} == {"RZ", "RX"}:
        return len(support_a & support_b) % 2 == 0

    raise ValueError(f"Unsupported operation pair: {op_a}, {op_b}")


def operation_can_be_moved_to_front(op_index, remaining_ops):
    """
    Check whether an operation can be executed before all earlier unexecuted
    operations.

    Parameters
    ----------
    op_index : int
        Index of the candidate operation in remaining_ops.

    remaining_ops : list[list]
        List of not-yet-executed logical operations.

    Returns
    -------
    bool
        True if the candidate operation commutes with every earlier operation.
    """
    op = remaining_ops[op_index]

    for previous_op in remaining_ops[:op_index]:
        if not logical_operations_commute(op, previous_op):
            return False

    return True


def is_single_body_logical_rz(op):
    """
    Check whether an operation is a single-body logical RZ rotation.

    Example
    -------
    ["RZ", [i], theta]
    """
    gate_type, wires, _ = op
    return gate_type.upper() == "RZ" and len(normalize_label(wires)) == 1


def is_two_body_logical_rz(op):
    """
    Check whether an operation is a two-body logical RZ parity rotation.

    Example
    -------
    ["RZ", [i, j], theta]
    """
    gate_type, wires, _ = op
    return gate_type.upper() == "RZ" and len(normalize_label(wires)) == 2


def cnot_depth(compiled_ops):
    """
    Compute the CNOT depth of a compiled circuit.

    Only CNOT gates are considered. Two CNOTs can be placed in the same CNOT
    layer if they act on disjoint physical qubits.

    Parameters
    ----------
    compiled_ops : list[list]
        Compiled physical operation list.

    Returns
    -------
    int
        Greedy CNOT depth.
    """
    wire_depth = {}

    for op in compiled_ops:
        gate_type, wires, _ = op
        gate_type = gate_type.upper()

        if gate_type != "CNOT":
            continue

        if len(wires) != 2:
            raise ValueError(f"CNOT must act on two wires, got {wires}.")

        c, t = wires

        layer = max(
            wire_depth.get(c, 0),
            wire_depth.get(t, 0),
        ) + 1

        wire_depth[c] = layer
        wire_depth[t] = layer

    return max(wire_depth.values(), default=0)


def single_qubit_depth(compiled_ops):
    """
    Compute the single-qubit gate depth of a compiled circuit.

    Only H, RX, and RZ gates are considered. Gates on different wires can be
    placed in the same single-qubit layer.

    Parameters
    ----------
    compiled_ops : list[list]
        Compiled physical operation list.

    Returns
    -------
    int
        Greedy single-qubit depth.
    """
    wire_depth = {}

    for op in compiled_ops:
        gate_type, wires, _ = op
        gate_type = gate_type.upper()

        if gate_type not in ("H", "RX", "RZ"):
            continue

        if len(wires) != 1:
            raise ValueError(
                f"{gate_type} should be a local single-qubit gate, got {wires}."
            )

        q = wires[0]

        layer = wire_depth.get(q, 0) + 1
        wire_depth[q] = layer

    return max(wire_depth.values(), default=0)


def total_greedy_depth(compiled_ops):
    """
    Compute the full greedy circuit depth including single-qubit gates and
    CNOT gates.

    Gates are processed in list order. Each gate is assigned to the earliest
    layer after all previous gates acting on its wires.

    Parameters
    ----------
    compiled_ops : list[list]
        Compiled physical operation list.

    Returns
    -------
    int
        Greedy total circuit depth.
    """
    wire_depth = {}

    for op in compiled_ops:
        gate_type, wires, _ = op
        gate_type = gate_type.upper()

        if gate_type in ("H", "RX", "RZ"):
            if len(wires) != 1:
                raise ValueError(
                    f"{gate_type} should be a local single-qubit gate, got {wires}."
                )

            q = wires[0]
            layer = wire_depth.get(q, 0) + 1
            wire_depth[q] = layer

        elif gate_type == "CNOT":
            if len(wires) != 2:
                raise ValueError(f"CNOT must act on two wires, got {wires}.")

            c, t = wires

            layer = max(
                wire_depth.get(c, 0),
                wire_depth.get(t, 0),
            ) + 1

            wire_depth[c] = layer
            wire_depth[t] = layer

        else:
            raise ValueError(f"Unsupported compiled operation type: {gate_type}")

    return max(wire_depth.values(), default=0)