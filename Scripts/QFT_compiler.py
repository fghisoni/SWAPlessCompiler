from Scripts.Compiler import (
    operation_physical_qubit,
    operation_can_be_moved_to_front,
    is_two_body_logical_rz,
    cnot_depth,
    single_qubit_depth,
    total_greedy_depth,
)

from Scripts.QFT_spanning_lines import generate_lhz_spanning_lines_left_to_right


def find_schedulable_local_operation(remaining_ops, current_line):
    """
    Find the next operation that can be scheduled on the current QFT spanning
    line.

    This function implements the QFT-specific scheduling policy.

    Rules
    -----
    - The first remaining operation is executed if it is local, regardless of
      whether it is H, RX, or RZ.
    - RX operations may be moved forward if:
        1. they are local in the current spanning line;
        2. they commute with all earlier unexecuted operations.
    - Two-body logical RZ rotations may be moved forward if:
        1. they are local in the current spanning line;
        2. they commute with all earlier unexecuted operations.
    - Single-body logical RZ rotations are not moved forward.
    - H gates are not moved forward.

    Parameters
    ----------
    remaining_ops : list[list]
        List of logical operations that have not yet been executed.

    current_line : list[tuple]
        Current parity-label spanning line.

    Returns
    -------
    tuple
        ``(idx, physical_qubit)``, where ``idx`` is the index of the selected
        operation in ``remaining_ops`` and ``physical_qubit`` is the wire on
        which it should be applied. If no operation can be scheduled, returns
        ``(None, None)``.
    """
    if not remaining_ops:
        return None, None

    # Execute the first operation if it is already local.
    first_op = remaining_ops[0]
    first_physical_qubit = operation_physical_qubit(first_op, current_line)

    if first_physical_qubit is not None:
        return 0, first_physical_qubit

    # Otherwise, only RX gates and two-body RZ parity rotations may commute
    # forward. Single-body RZ gates and H gates remain fixed in order.
    for idx, op in enumerate(remaining_ops[1:], start=1):
        gate_type = op[0].upper()

        allowed_to_move = (
            gate_type == "RX"
            or is_two_body_logical_rz(op)
        )

        if not allowed_to_move:
            continue

        physical_qubit = operation_physical_qubit(op, current_line)

        if physical_qubit is None:
            continue

        if operation_can_be_moved_to_front(idx, remaining_ops):
            return idx, physical_qubit

    return None, None


def compile_qft_with_lhz_spanning_lines(
    operations,
    n,
    spanning_data=None,
    verbose=False,
    return_debug=False,
    force_final_reversed=True,
):
    """
    Compile a QFT operation list using the precomputed LHZ left-to-right
    spanning-line schedule.

    The input operations should be expressed in terms of logical operations
    such as H, RX, and RZ parity rotations. The compiler walks along the QFT
    spanning-line sequence and applies an operation whenever it is local in the
    current parity frame. It may schedule RX gates and two-body RZ parity
    rotations earlier when this is allowed by commutation, but it keeps H gates
    and single-body RZ rotations fixed in order.

    Parameters
    ----------
    operations : list[list]
        Logical operation list. Supported gates are:
            ["H", [i], "N/A"]
            ["RX", [i], theta]
            ["RZ", [i], theta]
            ["RZ", [i, j], theta]

    n : int
        Number of logical qubits.

    spanning_data : dict or None
        Optional output of

            generate_lhz_spanning_lines_left_to_right(
                n,
                return_cnot_steps=True,
            )

        If None, the spanning-line data is generated automatically.

    verbose : bool
        If True, print placement and spanning-line transition information.

    return_debug : bool
        If True, return a debug dictionary in addition to the compiled circuit.

    force_final_reversed : bool
        If True, append the remaining CNOT transitions needed to end at the
        reversed QFT output spanning line.

    Returns
    -------
    compiled_ops : list[list]
        Physical operation list containing local single-qubit gates and CNOTs.

    debug : dict, optional
        Returned only if ``return_debug=True``. Contains placement information,
        final spanning line, gate counts, and depth estimates.
    """
    if spanning_data is None:
        spanning_data = generate_lhz_spanning_lines_left_to_right(
            n,
            return_cnot_steps=True,
            verbose=False,
        )

    label_lines = spanning_data["label_lines"]
    cnot_gates_by_update = spanning_data["cnot_gates_by_update"]

    if len(label_lines) == 0:
        raise ValueError("No spanning lines were generated.")

    remaining_ops = [list(op) for op in operations]

    compiled_ops = []
    placement_log = []

    line_index = 0
    current_line = label_lines[line_index]

    while remaining_ops:
        idx, physical_qubit = find_schedulable_local_operation(
            remaining_ops,
            current_line,
        )

        if idx is not None:
            logical_op = remaining_ops.pop(idx)
            gate_type, logical_wires, angle = logical_op
            gate_type = gate_type.upper()

            physical_op = [gate_type, [physical_qubit], angle]
            compiled_ops.append(physical_op)

            placement_log.append(
                {
                    "logical_operation": logical_op,
                    "physical_operation": physical_op,
                    "spanning_line_index": line_index,
                    "spanning_line": current_line,
                    "moved_across_previous_ops": idx > 0,
                    "remaining_index": idx,
                }
            )

            if verbose:
                moved_msg = " after commuting forward" if idx > 0 else ""
                print(
                    f"Placed {logical_op}{moved_msg} on line {line_index} "
                    f"{current_line} as {physical_op}"
                )

            continue

        # No currently local operation can be legally executed.
        # Move to the next QFT spanning line.
        if line_index >= len(label_lines) - 1:
            raise RuntimeError(
                "Reached the end of the QFT spanning-line schedule before all "
                "operations could be made local.\n"
                f"Remaining operations:\n"
                + "\n".join(str(op) for op in remaining_ops)
                + f"\nCurrent line: {current_line}\n"
                + f"Line index:   {line_index}"
            )

        transition_cnots = cnot_gates_by_update[line_index]
        compiled_ops.extend(transition_cnots)

        if verbose:
            print(
                f"Moved spanning line {line_index} -> {line_index + 1}: "
                f"{label_lines[line_index]} -> {label_lines[line_index + 1]}"
            )
            for cnot in transition_cnots:
                print("   ", cnot)

        line_index += 1
        current_line = label_lines[line_index]

    # Optionally continue along the QFT spanning-line schedule until reaching
    # the final reversed output line.
    if force_final_reversed:
        while line_index < len(label_lines) - 1:
            transition_cnots = cnot_gates_by_update[line_index]
            compiled_ops.extend(transition_cnots)

            if verbose:
                print(
                    f"Final move {line_index} -> {line_index + 1}: "
                    f"{label_lines[line_index]} -> {label_lines[line_index + 1]}"
                )
                for cnot in transition_cnots:
                    print("   ", cnot)

            line_index += 1
            current_line = label_lines[line_index]

    if return_debug:
        return compiled_ops, {
            "spanning_data": spanning_data,
            "placement_log": placement_log,
            "final_line_index": line_index,
            "final_spanning_line": current_line,
            "cnot_count": sum(1 for op in compiled_ops if op[0].upper() == "CNOT"),
            "rotation_count": sum(
                1 for op in compiled_ops if op[0].upper() in ("H", "RX", "RZ")
            ),
            "cnot_depth": cnot_depth(compiled_ops),
            "single_qubit_depth": single_qubit_depth(compiled_ops),
            "total_greedy_depth": total_greedy_depth(compiled_ops),
        }

    return compiled_ops