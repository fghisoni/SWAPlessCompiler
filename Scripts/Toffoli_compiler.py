# CnX_compiler.py

from itertools import combinations

from Scripts.LHZ import normalize_label, symdiff_labels
from Scripts.Compiler import (
    cnot_depth,
    single_qubit_depth,
    total_greedy_depth,
)
from Scripts.Toffoli_spanning_lines import generate_cnx_lhz_spanning_lines


def swap_as_cnots(q1, q2):
    """
    Decompose a nearest-neighbour SWAP(q1, q2) into three CNOTs.

    Parameters
    ----------
    q1, q2 : int
        Adjacent physical wires.

    Returns
    -------
    list[list]
        Gate list implementing SWAP(q1, q2) using CNOTs.
    """
    if abs(q1 - q2) != 1:
        raise ValueError(f"Only nearest-neighbour SWAPs are allowed, got {q1}, {q2}.")

    return [
        ["CNOT", [q1, q2], "N/A"],
        ["CNOT", [q2, q1], "N/A"],
        ["CNOT", [q1, q2], "N/A"],
    ]


def cnx_rz_physical_qubit(logical_wires, spanning_line):
    """
    Find the physical qubit carrying a given one- or two-body logical Z label.

    Parameters
    ----------
    logical_wires : list[int] or tuple[int]
        Logical support of the RZ rotation.

    spanning_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    int or None
        Physical qubit carrying the requested label, or None if unavailable.
    """
    target_label = normalize_label(logical_wires)

    matches = [
        q for q, label in enumerate(spanning_line)
        if normalize_label(label) == target_label
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def cnx_h_physical_qubit(logical_wires, spanning_line):
    """
    Find the physical qubit on which a logical Hadamard is local.

    The Hadamard is applied directly only when the required singleton label is
    present on the current spanning line.

    Parameters
    ----------
    logical_wires : list[int] or tuple[int]
        Logical support of the H operation. Must be a singleton.

    spanning_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    int or None
        Physical qubit carrying the requested singleton label, or None if
        unavailable.
    """
    target_label = normalize_label(logical_wires)

    if len(target_label) != 1:
        raise ValueError(f"H must act on one logical wire, got {logical_wires}.")

    matches = [
        q for q, label in enumerate(spanning_line)
        if normalize_label(label) == target_label
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def split_cnx_operations(operations):
    """
    Split the logical C^nX operation list into the initial H, the RZ block,
    and the final H.

    The expected input is the output of generate_cnx_logical_z_gates:

        ["H", [target], "N/A"]
        ["RZ", support, theta]
        ...
        ["H", [target], "N/A"]

    Parameters
    ----------
    operations : list[list]
        Logical operation list.

    Returns
    -------
    initial_h : list
        First Hadamard operation.

    rz_ops : list[list]
        Middle logical RZ operations.

    final_h : list
        Final Hadamard operation.
    """
    if len(operations) < 3:
        raise ValueError("C^nX operation list is too short.")

    initial_h = list(operations[0])
    final_h = list(operations[-1])
    middle_ops = operations[1:-1]

    if initial_h[0].upper() != "H":
        raise ValueError(f"Expected first operation to be H, got {initial_h}.")

    if final_h[0].upper() != "H":
        raise ValueError(f"Expected last operation to be H, got {final_h}.")

    rz_ops = []

    for op in middle_ops:
        op = list(op)

        if op[0].upper() != "RZ":
            raise ValueError(f"Expected middle operations to be RZ, got {op}.")

        rz_ops.append(op)

    return initial_h, rz_ops, final_h


def find_local_low_body_rz(rz_ops, current_line):
    """
    Find the first one- or two-body RZ operation that is local on the current
    spanning line.

    Parameters
    ----------
    rz_ops : list[list]
        Remaining logical RZ operations.

    current_line : list[tuple]
        Current spanning line.

    Returns
    -------
    tuple[int | None, int | None]
        Operation index and physical qubit. If no operation is local, returns
        (None, None).
    """
    for idx, op in enumerate(rz_ops):
        _, wires, _ = op
        support = normalize_label(wires)

        if len(support) > 2:
            continue

        physical_qubit = cnx_rz_physical_qubit(support, current_line)

        if physical_qubit is not None:
            return idx, physical_qubit

    return None, None


def labels_to_layout(spanning_line):
    """
    Convert a spanning line into a mutable physical layout.

    The layout is a list such that layout[q] is the logical parity label stored
    at physical wire q.

    Parameters
    ----------
    spanning_line : list[tuple]
        Spanning line as parity labels.

    Returns
    -------
    list[tuple]
        Mutable physical layout.
    """
    return [normalize_label(label) for label in spanning_line]


def find_label_subset_for_support_at_final_line(support, layout, ancilla_wire):
    """
    Find physical positions whose current parity labels have symmetric
    difference equal to the requested logical support.

    This is needed because the final spanning line may contain parity labels
    rather than singleton labels.

    Example
    -------
    If

        support = (0, 1, 2)
        layout  = [(1, 2), (0, 2), (2,), (3,)]

    then this function may return

        [0, 1, 2]

    since

        (1, 2) Δ (0, 2) Δ (2,) = (0, 1, 2).

    Parameters
    ----------
    support : list[int] or tuple[int]
        Logical support of the many-body RZ rotation.

    layout : list[tuple]
        Current physical layout, where layout[q] is the parity label stored on
        physical wire q.

    ancilla_wire : int
        Physical wire of the clean ancilla.

    Returns
    -------
    list[int]
        Physical positions whose labels generate the requested support.
    """
    target_support = normalize_label(support)

    candidate_positions = []
    candidate_labels = []

    for q, label in enumerate(layout):
        if q == ancilla_wire:
            continue

        label = normalize_label(label)

        # Do not use the ancilla label as part of the logical support.
        if label == (ancilla_wire,):
            continue

        candidate_positions.append(q)
        candidate_labels.append(label)

    best_positions = None
    best_cost = None

    # Search from small subsets to large subsets.
    for subset_size in range(1, len(candidate_positions) + 1):
        for idx_subset in combinations(range(len(candidate_positions)), subset_size):
            labels = [candidate_labels[i] for i in idx_subset]

            if symdiff_labels(labels) != target_support:
                continue

            positions = [candidate_positions[i] for i in idx_subset]

            # Prefer subsets that are already close to the ancilla.
            cost = sum(abs(q - ancilla_wire) for q in positions)

            if best_positions is None or cost < best_cost:
                best_positions = positions
                best_cost = cost

        if best_positions is not None:
            return sorted(best_positions)

    raise ValueError(
        "Could not express the requested many-body support as a symmetric "
        "difference of labels in the final spanning line.\n"
        f"Support: {target_support}\n"
        f"Layout:  {layout}"
    )


def pack_positions_next_to_ancilla(selected_positions, layout, ancilla_wire):
    """
    Move selected physical positions into a contiguous block immediately to the
    left of the ancilla using nearest-neighbour SWAPs.

    The selected positions are tracked by physical position rather than by
    singleton logical labels. This is necessary because the final spanning line
    may contain parity labels such as (1,2) and (0,2), rather than only
    singleton labels.

    Parameters
    ----------
    selected_positions : list[int]
        Physical positions whose labels generate the required many-body support.

    layout : list[tuple]
        Current physical layout.

    ancilla_wire : int
        Fixed physical wire of the clean ancilla.

    Returns
    -------
    swap_ops : list[list]
        CNOT gates implementing the required SWAPs.

    undo_swap_ops : list[list]
        Reverse SWAP sequence, also decomposed into CNOTs.

    packed_layout : list[tuple]
        Layout after packing.

    active_block : list[int]
        Contiguous physical block immediately to the left of the ancilla.
    """
    selected_positions = sorted(selected_positions)
    m = len(selected_positions)

    if m < 2:
        raise ValueError("At least two positions are required for parity packing.")

    if ancilla_wire != len(layout) - 1:
        raise ValueError(
            "This routine assumes the ancilla is fixed at the right boundary."
        )

    if any(q >= ancilla_wire for q in selected_positions):
        raise ValueError(
            "Selected logical positions must lie to the left of the ancilla.\n"
            f"Selected positions: {selected_positions}\n"
            f"Ancilla wire:       {ancilla_wire}"
        )

    target_slots = list(range(ancilla_wire - m, ancilla_wire))

    packed_layout = list(layout)
    selected = [False] * len(layout)

    for q in selected_positions:
        selected[q] = True

    swap_ops = []
    undo_swap_ops = []

    # Fill target slots from right to left.
    for slot in reversed(target_slots):
        selected_to_left = [q for q in range(slot + 1) if selected[q]]

        if not selected_to_left:
            raise RuntimeError(
                "Could not find a selected label to move into the target slot.\n"
                f"Selected positions: {selected_positions}\n"
                f"Target slots:       {target_slots}\n"
                f"Current selected:   {selected}"
            )

        pos = max(selected_to_left)

        # Bubble the selected label to the target slot.
        for q in range(pos, slot):
            swap_gates = swap_as_cnots(q, q + 1)

            swap_ops.extend(swap_gates)
            undo_swap_ops = swap_gates + undo_swap_ops

            packed_layout[q], packed_layout[q + 1] = (
                packed_layout[q + 1],
                packed_layout[q],
            )
            selected[q], selected[q + 1] = selected[q + 1], selected[q]

    active_block = target_slots

    return swap_ops, undo_swap_ops, packed_layout, active_block


def parity_chain_to_ancilla(active_block, ancilla_wire):
    """
    Generate the nearest-neighbour CNOT chain that computes the parity of a
    contiguous active block into the ancilla.

    Example
    -------
    If

        active_block = [1, 2, 3]
        ancilla_wire = 4

    this returns

        CNOT(1, 2), CNOT(2, 3), CNOT(3, 4).

    Parameters
    ----------
    active_block : list[int]
        Contiguous physical wires immediately to the left of the ancilla.

    ancilla_wire : int
        Physical wire of the clean ancilla.

    Returns
    -------
    list[list]
        Nearest-neighbour CNOT chain.
    """
    if not active_block:
        raise ValueError("active_block cannot be empty.")

    if active_block[-1] != ancilla_wire - 1:
        raise ValueError(
            "The active block must end immediately to the left of the ancilla.\n"
            f"Active block: {active_block}\n"
            f"Ancilla:     {ancilla_wire}"
        )

    for q1, q2 in zip(active_block[:-1], active_block[1:]):
        if q2 != q1 + 1:
            raise ValueError(f"active_block is not contiguous: {active_block}")

    gates = []

    for q in active_block:
        gates.append(["CNOT", [q, q + 1], "N/A"])

    return gates


def compile_many_body_rz_at_final_line(operation, final_layout, ancilla_wire):
    """
    Compile a many-body logical RZ rotation at the final spanning line.

    Instead of assuming that every logical qubit appears as a singleton label,
    this routine finds a subset of the current parity labels whose symmetric
    difference equals the desired support.

    It then moves those physical labels next to the fixed ancilla, computes
    their parity into the ancilla using a nearest-neighbour CNOT chain, applies
    RZ on the ancilla, and uncomputes.

    Parameters
    ----------
    operation : list
        Logical operation ["RZ", support, theta] with len(support) > 2.

    final_layout : list[tuple]
        Final physical layout.

    ancilla_wire : int
        Physical wire of the clean ancilla.

    Returns
    -------
    list[list]
        Physical gate list.
    """
    gate_type, support, theta = operation
    gate_type = gate_type.upper()

    if gate_type != "RZ":
        raise ValueError(f"Expected RZ operation, got {operation}.")

    support = normalize_label(support)

    if len(support) <= 2:
        raise ValueError(
            "compile_many_body_rz_at_final_line only accepts supports of size > 2."
        )

    selected_positions = find_label_subset_for_support_at_final_line(
        support=support,
        layout=final_layout,
        ancilla_wire=ancilla_wire,
    )

    (
        swap_ops,
        undo_swap_ops,
        packed_layout,
        active_block,
    ) = pack_positions_next_to_ancilla(
        selected_positions=selected_positions,
        layout=final_layout,
        ancilla_wire=ancilla_wire,
    )

    collect_chain = parity_chain_to_ancilla(active_block, ancilla_wire)
    uncompute_chain = list(reversed(collect_chain))

    compiled = []
    compiled.extend(swap_ops)
    compiled.extend(collect_chain)
    compiled.append(["RZ", [ancilla_wire], theta])
    compiled.extend(uncompute_chain)
    compiled.extend(undo_swap_ops)

    return compiled


def compile_cnx_with_lhz_spanning_lines(
    operations,
    n_controls,
    spanning_data=None,
    verbose=False,
    return_debug=False,
):
    """
    Compile a C^nX gate using LHZ spanning lines for the one- and two-body
    logical Z rotations, and a final-line SWAP-based routine for the remaining
    many-body rotations.

    Wire convention
    ---------------
    The compiler assumes

        controls = [0, 1, ..., n_controls - 1]
        target   = n_controls
        ancilla  = n_controls + 1

    The input operation list should be generated by

        generate_cnx_logical_z_gates(n_controls)

    and contains H gates on the target and logical RZ rotations on arbitrary
    supports.

    Compilation strategy
    --------------------
    1. Apply the initial Hadamard on the target.
    2. Move through the LHZ spanning lines and implement all one- and two-body
       RZ rotations as soon as their labels become available.
    3. Continue until the final spanning line of the C^nX schedule is reached.
    4. Implement all remaining many-body RZ rotations by expressing their
       support as a symmetric difference of labels on the final line, packing
       those labels next to the right-boundary ancilla, applying an RZ on the
       ancilla, and uncomputing.
    5. Apply the final Hadamard on the target.

    Parameters
    ----------
    operations : list[list]
        Output of generate_cnx_logical_z_gates.

    n_controls : int
        Number of control qubits.

    spanning_data : dict or None
        Optional output of generate_cnx_lhz_spanning_lines(...,
        return_cnot_steps=True).

    verbose : bool
        If True, print compilation information.

    return_debug : bool
        If True, return a debug dictionary.

    Returns
    -------
    compiled_ops : list[list]
        Physical gate list.

    debug : dict, optional
        Returned only if return_debug=True.
    """
    if n_controls < 1:
        raise ValueError("n_controls must be at least 1.")

    target = n_controls
    ancilla = n_controls + 1

    if spanning_data is None:
        spanning_data = generate_cnx_lhz_spanning_lines(
            n_controls,
            return_cnot_steps=True,
            verbose=False,
        )

    label_lines = spanning_data["label_lines"]
    cnot_gates_by_update = spanning_data["cnot_gates_by_update"]

    if len(label_lines) == 0:
        raise ValueError("No C^nX spanning lines were generated.")

    initial_h, rz_ops, final_h = split_cnx_operations(operations)

    low_body_rz = []
    many_body_rz = []

    for op in rz_ops:
        support = normalize_label(op[1])

        if len(support) <= 2:
            low_body_rz.append(op)
        else:
            many_body_rz.append(op)

    compiled_ops = []
    placement_log = []

    line_index = 0
    current_line = label_lines[line_index]

    # ---------------------------------------------------------------------
    # 1. Initial Hadamard
    # ---------------------------------------------------------------------
    h_physical = cnx_h_physical_qubit(initial_h[1], current_line)

    if h_physical is None:
        raise RuntimeError(
            "Initial H is not local on the first C^nX spanning line.\n"
            f"Initial H:    {initial_h}\n"
            f"Current line: {current_line}"
        )

    physical_h = ["H", [h_physical], "N/A"]
    compiled_ops.append(physical_h)

    placement_log.append(
        {
            "logical_operation": initial_h,
            "physical_operation": physical_h,
            "spanning_line_index": line_index,
            "spanning_line": current_line,
            "moved_across_previous_ops": False,
            "remaining_index": 0,
        }
    )

    if verbose:
        print(f"Placed initial H as {physical_h}")

    # ---------------------------------------------------------------------
    # 2. Sweep through spanning lines and place one-/two-body RZ rotations.
    # ---------------------------------------------------------------------
    while True:
        placed_any = True

        while placed_any:
            placed_any = False

            op_index, physical_qubit = find_local_low_body_rz(
                low_body_rz,
                current_line,
            )

            if op_index is None:
                break

            logical_op = low_body_rz.pop(op_index)
            _, _, angle = logical_op

            physical_op = ["RZ", [physical_qubit], angle]
            compiled_ops.append(physical_op)

            placement_log.append(
                {
                    "logical_operation": logical_op,
                    "physical_operation": physical_op,
                    "spanning_line_index": line_index,
                    "spanning_line": current_line,
                    "moved_across_previous_ops": True,
                    "remaining_index": op_index,
                }
            )

            if verbose:
                print(
                    f"Placed {logical_op} on line {line_index} "
                    f"{current_line} as {physical_op}"
                )

            placed_any = True

        # Continue until the final available spanning line.
        if line_index >= len(label_lines) - 1:
            break

        transition_cnots = cnot_gates_by_update[line_index]
        compiled_ops.extend(transition_cnots)

        if verbose:
            print(
                f"Moved C^nX line {line_index} -> {line_index + 1}: "
                f"{label_lines[line_index]} -> {label_lines[line_index + 1]}"
            )
            for cnot in transition_cnots:
                print("   ", cnot)

        line_index += 1
        current_line = label_lines[line_index]

    if low_body_rz:
        raise RuntimeError(
            "Some one- or two-body RZ rotations were not implemented during "
            "the spanning-line sweep:\n"
            + "\n".join(str(op) for op in low_body_rz)
            + f"\nFinal line: {current_line}"
        )

    # ---------------------------------------------------------------------
    # 3. Compile remaining many-body RZ rotations at the final line.
    # ---------------------------------------------------------------------
    final_layout = labels_to_layout(current_line)

    for logical_op in many_body_rz:
        many_body_ops = compile_many_body_rz_at_final_line(
            logical_op,
            final_layout,
            ancilla,
        )

        compiled_ops.extend(many_body_ops)

        for physical_op in many_body_ops:
            if physical_op[0].upper() == "RZ":
                placement_log.append(
                    {
                        "logical_operation": logical_op,
                        "physical_operation": physical_op,
                        "spanning_line_index": line_index,
                        "spanning_line": current_line,
                        "moved_across_previous_ops": True,
                        "remaining_index": None,
                    }
                )

        if verbose:
            print(f"Placed many-body {logical_op} at final line:")
            for op in many_body_ops:
                print("   ", op)

    # ---------------------------------------------------------------------
    # 4. Final Hadamard
    # ---------------------------------------------------------------------
    h_physical = cnx_h_physical_qubit(final_h[1], current_line)

    if h_physical is None:
        raise RuntimeError(
            "Final H is not local on the final C^nX spanning line.\n"
            f"Final H:      {final_h}\n"
            f"Final line:   {current_line}"
        )

    physical_h = ["H", [h_physical], "N/A"]
    compiled_ops.append(physical_h)

    placement_log.append(
        {
            "logical_operation": final_h,
            "physical_operation": physical_h,
            "spanning_line_index": line_index,
            "spanning_line": current_line,
            "moved_across_previous_ops": False,
            "remaining_index": None,
        }
    )

    if verbose:
        print(f"Placed final H as {physical_h}")

    if return_debug:
        return compiled_ops, {
            "unitary": "C^nX",
            "spanning_data": spanning_data,
            "placement_log": placement_log,
            "final_line_index": line_index,
            "final_spanning_line": current_line,
            "target": target,
            "ancilla": ancilla,
            "cnot_count": sum(1 for op in compiled_ops if op[0].upper() == "CNOT"),
            "rotation_count": sum(
                1 for op in compiled_ops if op[0].upper() in ("H", "RX", "RZ")
            ),
            "many_body_count": len(many_body_rz),
            "cnot_depth": cnot_depth(compiled_ops),
            "single_qubit_depth": single_qubit_depth(compiled_ops),
            "total_greedy_depth": total_greedy_depth(compiled_ops),
        }

    return compiled_ops