from Scripts.QFT_spanning_lines import generate_lhz_spanning_lines_left_to_right

from Scripts.Spanning_lines import (
    get_label,
    canonical_state,
    labels_from_state,
    cnot_steps_from_moves,
    cnot_gate_list_from_steps,
)


def generate_cnx_lhz_spanning_lines(
    n_controls,
    return_cnot_steps=False,
    verbose=False,
):
    """
    Generate the LHZ spanning lines used for the C^nX decomposition.

    Wire convention
    ---------------
    The C^nX construction uses:

        controls = [0, 1, ..., n_controls - 1]
        target   = n_controls
        ancilla  = n_controls + 1

    The logical C^nZ part acts on the controls plus the target, so the number
    of logical qubits passed to the QFT-style LHZ spanning-line generator is

        n_logical = n_controls + 1.

    The clean ancilla is not part of the LHZ parity-label tracking. After
    generating each spanning line for the controls and target, the ancilla
    singleton label is appended to the line.

    Stopping rule
    -------------
    The function generates QFT-style spanning lines until the step immediately
    before the first line whose first label would become

        (n_controls - 1, n_controls).

    It then performs a modified final update: from the previous line, it applies
    all right-moving stabilizer moves except those acting on rows whose current
    label contains the target value n_controls.

    This means that the final line has first label

        (n_controls - 1, n_controls),

    but labels involving the target that should remain fixed are not updated.

    For n_controls = 2, the output is

        [(0,),   (1,),   (2,), (3,)]
        [(0, 1), (1,),   (2,), (3,)]
        [(0, 1), (0, 2), (2,), (3,)]
        [(1, 2), (0, 2), (2,), (3,)]

    Parameters
    ----------
    n_controls : int
        Number of control qubits in the C^nX gate.

    return_cnot_steps : bool
        If False, return only the label spanning lines.

        If True, return a dictionary in the same style as
        generate_lhz_spanning_lines_left_to_right, truncated according to the
        modified stopping rule and with the ancilla appended to each label line.

    verbose : bool
        If True, print the generated spanning lines.

    Returns
    -------
    list or dict
        If return_cnot_steps=False, returns a list of label spanning lines.

        If return_cnot_steps=True, returns a dictionary containing:
            - "unitary"
            - "label_lines"
            - "coord_lines"
            - "step_moves"
            - "cnot_steps_by_update"
            - "cnot_gates_by_update"
            - "frozen_rows_by_update"
            - "grid"
            - "faces"
            - "ancilla"
            - "target"
            - "stop_label"
            - "stop_index"
    """
    if n_controls < 1:
        raise ValueError("n_controls must be at least 1.")

    n_logical = n_controls + 1
    target = n_controls
    ancilla = n_controls + 1

    stop_label = (n_controls - 1, n_controls)

    qft_data = generate_lhz_spanning_lines_left_to_right(
        n_logical,
        return_cnot_steps=True,
        verbose=False,
    )

    grid = qft_data["grid"]
    raw_label_lines_full = qft_data["label_lines"]
    raw_coord_lines_full = qft_data["coord_lines"]

    stop_index = None
    for idx, line in enumerate(raw_label_lines_full):
        if tuple(line[0]) == stop_label:
            stop_index = idx
            break

    if stop_index is None:
        raise RuntimeError(
            "Could not find the requested C^nX stopping line in the QFT-style "
            "spanning-line schedule.\n"
            f"Requested first label: {stop_label}\n"
            f"Available first labels: {[line[0] for line in raw_label_lines_full]}"
        )

    if stop_index == 0:
        raise RuntimeError(
            "The stopping line is the initial line, so there is no previous line "
            "from which to construct the modified final update."
        )

    # Keep all lines before the full QFT stopping line.
    raw_coord_lines = list(raw_coord_lines_full[:stop_index])
    raw_label_lines = list(raw_label_lines_full[:stop_index])

    # Keep all complete transitions before the final modified transition.
    step_moves = list(qft_data["step_moves"][: stop_index - 1])
    cnot_steps_by_update = list(qft_data["cnot_steps_by_update"][: stop_index - 1])
    cnot_gates_by_update = list(qft_data["cnot_gates_by_update"][: stop_index - 1])

    # Build the modified final transition from the previous line.
    previous_state = raw_coord_lines_full[stop_index - 1]
    candidate_moves = qft_data["step_moves"][stop_index - 1]

    filtered_final_moves = []

    for move in candidate_moves:
        row = move["row"]
        old_coord = previous_state[row]
        old_label = get_label(grid, old_coord)

        # Do not update rows whose current label contains the target n_controls.
        if target in old_label:
            continue

        filtered_final_moves.append(move)

    if not filtered_final_moves:
        raise RuntimeError(
            "The modified final update selected no stabilizer moves.\n"
            f"Previous line: {labels_from_state(previous_state, grid)}\n"
            f"Target value excluded from moving labels: {target}"
        )

    final_state = list(previous_state)

    for move in filtered_final_moves:
        row = move["row"]
        final_state[row] = move["new_coord"]

    final_state = canonical_state(tuple(final_state), grid)
    final_labels = labels_from_state(final_state, grid)

    if tuple(final_labels[0]) != stop_label:
        raise RuntimeError(
            "Modified final update did not produce the requested first label.\n"
            f"Expected first label: {stop_label}\n"
            f"Actual final labels:  {final_labels}"
        )

    final_cnot_steps = cnot_steps_from_moves(filtered_final_moves)
    final_cnot_gates = cnot_gate_list_from_steps(final_cnot_steps)

    raw_coord_lines.append(final_state)
    raw_label_lines.append(final_labels)

    step_moves.append(filtered_final_moves)
    cnot_steps_by_update.append(final_cnot_steps)
    cnot_gates_by_update.append(final_cnot_gates)

    frozen_rows_by_update = qft_data.get("frozen_rows_by_update", None)
    if frozen_rows_by_update is not None:
        frozen_rows_by_update = frozen_rows_by_update[:stop_index]
        frozen_rows_by_update.append(frozen_rows_by_update[-1])

    # Append the clean ancilla as an extra singleton label.
    label_lines = [
        list(line) + [(ancilla,)]
        for line in raw_label_lines
    ]

    if verbose:
        for step, line in enumerate(label_lines):
            marker = ""
            if step == len(label_lines) - 1:
                marker = "  <-- modified final line"
            print(f"C^nX step {step}: {line}{marker}")

    if return_cnot_steps:
        return {
            "unitary": "C^nX",
            "label_lines": label_lines,
            "coord_lines": raw_coord_lines,
            "step_moves": step_moves,
            "cnot_steps_by_update": cnot_steps_by_update,
            "cnot_gates_by_update": cnot_gates_by_update,
            "frozen_rows_by_update": frozen_rows_by_update,
            "grid": qft_data["grid"],
            "faces": qft_data["faces"],
            "ancilla": ancilla,
            "target": target,
            "stop_label": stop_label,
            "stop_index": stop_index,
        }

    return label_lines