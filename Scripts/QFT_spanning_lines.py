from Scripts.LHZ import build_extended_lhz_array
from Scripts.Spanning_lines import (
    get_label,
    canonical_state,
    labels_from_state,
    is_valid_spanning_line_coords,
    build_lhz_faces,
    right_moving_face_crossings,
    cnot_steps_from_moves,
    cnot_gate_list_from_steps,
)


def target_label_for_row(row: int, n: int):
    """
    Return the target label for a row in the QFT reversed-output line.

    For QFT, the final spanning line is the reversed singleton line,

        [(n-1,), (n-2,), ..., (0,)].

    Therefore row ``i`` has target label ``(n-1-i,)``.

    Parameters
    ----------
    row : int
        Row index.

    n : int
        Number of logical qubits.

    Returns
    -------
    tuple
        Singleton parity label for the target output row.
    """
    return (n - 1 - row,)


def row_has_target_label(state, row: int, n: int, grid):
    """
    Check whether a row of a coordinate spanning line has reached its QFT
    reversed-output target label.

    Parameters
    ----------
    state : tuple
        Coordinate spanning line.

    row : int
        Row index.

    n : int
        Number of logical qubits.

    grid : np.ndarray
        Extended LHZ grid.

    Returns
    -------
    bool
        True if the label in the row equals ``(n-1-row,)``.
    """
    return get_label(grid, state[row]) == target_label_for_row(row, n)


def is_reversed_output_line(state, n: int, grid):
    """
    Check whether a coordinate spanning line equals the QFT reversed-output
    line.

    Parameters
    ----------
    state : tuple
        Coordinate spanning line.

    n : int
        Number of logical qubits.

    grid : np.ndarray
        Extended LHZ grid.

    Returns
    -------
    bool
        True if the labels are ``[(n-1,), ..., (0,)]``.
    """
    return all(
        row_has_target_label(state, row, n, grid)
        for row in range(n)
    )


def initial_diagonal_spanning_line(n: int, grid):
    """
    Return the initial data-qubit diagonal spanning line for the QFT sweep.

    For the extended LHZ grid used here, the initial line is

        [(0, 0), (1, 1), ..., (n-1, n-1)]

    and should carry labels

        [(0,), (1,), ..., (n-1,)].

    Parameters
    ----------
    n : int
        Number of logical qubits.

    grid : np.ndarray
        Extended LHZ grid.

    Returns
    -------
    tuple
        Coordinate representation of the initial spanning line.

    Raises
    ------
    ValueError
        If the diagonal line is not a valid spanning line or does not carry
        the expected singleton labels.
    """
    line = tuple((row, row) for row in range(n))
    line = canonical_state(line, grid)

    if not is_valid_spanning_line_coords(list(line), grid):
        raise ValueError(
            "The default diagonal coordinate line is not a valid spanning line."
        )

    expected_labels = [(i,) for i in range(n)]
    actual_labels = labels_from_state(line, grid)

    if actual_labels != expected_labels:
        raise ValueError(
            "The default diagonal line does not correspond to the data-qubit line.\n"
            f"Expected labels: {expected_labels}\n"
            f"Actual labels:   {actual_labels}"
        )

    return line


def choose_compatible_right_moves_with_freezing(
    state,
    moves,
    n: int,
    grid,
    frozen_rows,
):
    """
    Choose compatible right-moving face crossings while enforcing QFT row
    freezing.

    Rows are not frozen just because they currently have their final label.
    Instead, a row is frozen only after it moves into its final reversed-output
    target label during the sweep. This is important for odd ``n``, where the
    middle row starts with the same singleton label that it must have at the
    end, but still needs to move temporarily.

    Parameters
    ----------
    state : tuple
        Current coordinate spanning line.

    moves : list[dict]
        Candidate right-moving moves, typically returned by
        ``right_moving_face_crossings``.

    n : int
        Number of logical qubits.

    grid : np.ndarray
        Extended LHZ grid.

    frozen_rows : set[int]
        Rows that have already reached their target label and should no longer
        move.

    Returns
    -------
    tuple
        ``(chosen_moves, new_state, new_frozen_rows)``.
    """
    if not moves:
        return [], state, set(frozen_rows)

    moves_by_row = {}

    for move in moves:
        row = move["row"]

        if row in frozen_rows:
            continue

        if row not in moves_by_row:
            moves_by_row[row] = move

    candidate_state = list(state)
    chosen_moves = []
    new_frozen_rows = set(frozen_rows)

    for row in sorted(moves_by_row):
        move = moves_by_row[row]

        trial_state = list(candidate_state)
        old_label = get_label(grid, trial_state[row])

        trial_state[row] = move["new_coord"]
        trial_state = canonical_state(tuple(trial_state), grid)

        new_label = get_label(grid, trial_state[row])

        if is_valid_spanning_line_coords(list(trial_state), grid):
            candidate_state = list(trial_state)
            chosen_moves.append(move)

            if (
                old_label != target_label_for_row(row, n)
                and new_label == target_label_for_row(row, n)
            ):
                new_frozen_rows.add(row)

    new_state = canonical_state(tuple(candidate_state), grid)

    return chosen_moves, new_state, new_frozen_rows


def generate_lhz_spanning_lines_left_to_right(
    n: int,
    max_steps=None,
    return_coords=False,
    return_cnot_steps=False,
    verbose=False,
):
    """
    Generate the QFT left-to-right spanning-line schedule in the extended LHZ
    architecture.

    The schedule starts from the initial data-qubit line

        [(0,), (1,), ..., (n-1,)]

    and repeatedly applies compatible right-moving triangle/plaquette crossings.
    When a row moves into its final reversed-output singleton label, that row
    is frozen. The process terminates when the full reversed-output line

        [(n-1,), (n-2,), ..., (0,)]

    is reached.

    Parameters
    ----------
    n : int
        Number of logical qubits.

    max_steps : int or None
        Maximum number of global update steps. If None, a conservative default
        is used.

    return_coords : bool
        If True, return both label lines and coordinate lines.

    return_cnot_steps : bool
        If True, return a dictionary containing label lines, coordinate lines,
        move metadata, CNOT metadata, and the underlying grid/faces.

    verbose : bool
        If True, print the generated spanning lines and frozen rows.

    Returns
    -------
    list or tuple or dict
        If ``return_cnot_steps=False`` and ``return_coords=False``, returns only
        the label spanning lines.

        If ``return_coords=True``, returns ``(label_lines, coord_lines)``.

        If ``return_cnot_steps=True``, returns a dictionary with:
            - ``"label_lines"``;
            - ``"coord_lines"``;
            - ``"step_moves"``;
            - ``"cnot_steps_by_update"``;
            - ``"cnot_gates_by_update"``;
            - ``"frozen_rows_by_update"``;
            - ``"grid"``;
            - ``"faces"``.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    grid = build_extended_lhz_array(n)
    faces = build_lhz_faces(grid)

    if max_steps is None:
        max_steps = 4 * n * (n + 1)

    current_state = initial_diagonal_spanning_line(n, grid)

    frozen_rows = set()

    coord_lines = [current_state]
    label_lines = [labels_from_state(current_state, grid)]

    step_moves = []
    cnot_steps_by_update = []
    cnot_gates_by_update = []
    frozen_rows_by_update = [set(frozen_rows)]

    if verbose:
        print("step 0:", label_lines[-1], "frozen:", sorted(frozen_rows))

    for step in range(1, max_steps + 1):
        if is_reversed_output_line(current_state, n, grid):
            break

        moves = right_moving_face_crossings(current_state, grid, faces)

        chosen_moves, new_state, frozen_rows = (
            choose_compatible_right_moves_with_freezing(
                current_state,
                moves,
                n,
                grid,
                frozen_rows,
            )
        )

        if not chosen_moves:
            raise RuntimeError(
                "The QFT left-to-right sweep got stuck before reaching the "
                "reversed output line.\n"
                f"Current labels: {labels_from_state(current_state, grid)}\n"
                f"Target labels:  {[(i,) for i in reversed(range(n))]}\n"
                f"Frozen rows:    {sorted(frozen_rows)}\n"
                f"Current coords: {current_state}\n"
            )

        cnot_steps = cnot_steps_from_moves(chosen_moves)
        cnot_gates = cnot_gate_list_from_steps(cnot_steps)

        current_state = new_state

        coord_lines.append(current_state)
        label_lines.append(labels_from_state(current_state, grid))
        step_moves.append(chosen_moves)
        cnot_steps_by_update.append(cnot_steps)
        cnot_gates_by_update.append(cnot_gates)
        frozen_rows_by_update.append(set(frozen_rows))

        if verbose:
            print(
                f"step {step}:",
                label_lines[-1],
                "frozen:",
                sorted(frozen_rows),
            )

        if is_reversed_output_line(current_state, n, grid):
            break

    if not is_reversed_output_line(current_state, n, grid):
        raise RuntimeError(
            "Maximum number of steps reached before the reversed output line "
            "was obtained.\n"
            f"Last labels:   {labels_from_state(current_state, grid)}\n"
            f"Target labels: {[(i,) for i in reversed(range(n))]}\n"
            f"Frozen rows:   {sorted(frozen_rows)}"
        )

    if return_cnot_steps:
        return {
            "label_lines": label_lines,
            "coord_lines": coord_lines,
            "step_moves": step_moves,
            "cnot_steps_by_update": cnot_steps_by_update,
            "cnot_gates_by_update": cnot_gates_by_update,
            "frozen_rows_by_update": frozen_rows_by_update,
            "grid": grid,
            "faces": faces,
        }

    if return_coords:
        return label_lines, coord_lines

    return label_lines