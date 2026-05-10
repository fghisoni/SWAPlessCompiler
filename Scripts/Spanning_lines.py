import numpy as np
from collections import deque

from Scripts.LHZ import normalize_label, symdiff_labels, build_extended_lhz_array


def periodic_col(col: int, width: int) -> int:
    """
    Return the column index obtained by imposing horizontal periodic boundary
    conditions.

    Parameters
    ----------
    col : int
        Column index before wrapping.

    width : int
        Number of columns in the extended LHZ grid.

    Returns
    -------
    int
        Wrapped column index.
    """
    return col % width


def periodic_distance_is_one(c1: int, c2: int, width: int) -> bool:
    """
    Check whether two column indices are nearest neighbours under horizontal
    periodic boundary conditions.

    Parameters
    ----------
    c1, c2 : int
        Column indices.

    width : int
        Number of columns in the extended LHZ grid.

    Returns
    -------
    bool
        True if the columns differ by one modulo the horizontal period.
    """
    return (c1 - c2) % width in (1, width - 1)


def is_valid_node(grid: np.ndarray, coord) -> bool:
    """
    Check whether a coordinate corresponds to an actual node of the extended
    LHZ grid.

    Parameters
    ----------
    grid : np.ndarray
        Extended LHZ grid.

    coord : tuple[int, int]
        Coordinate of the form ``(row, col)``.

    Returns
    -------
    bool
        True if the coordinate lies in a valid row and the corresponding grid
        entry is not ``None``.
    """
    r, c = coord
    n_rows, n_cols = grid.shape

    if not (0 <= r < n_rows):
        return False

    c = periodic_col(c, n_cols)
    return grid[r, c] is not None


def get_label(grid: np.ndarray, coord):
    """
    Return the parity label stored at a coordinate of the extended LHZ grid.

    Horizontal periodic boundary conditions are applied to the column index.

    Parameters
    ----------
    grid : np.ndarray
        Extended LHZ grid.

    coord : tuple[int, int]
        Coordinate of the form ``(row, col)``.

    Returns
    -------
    tuple
        Parity label stored at the coordinate.

    Raises
    ------
    ValueError
        If the coordinate does not correspond to a valid node.
    """
    r, c = coord
    c = periodic_col(c, grid.shape[1])

    label = grid[r, c]
    if label is None:
        raise ValueError(f"Coordinate {coord} is not a valid LHZ node.")

    return label


def canonical_coord(grid: np.ndarray, coord):
    """
    Canonicalize a coordinate by wrapping its column index periodically.

    Parameters
    ----------
    grid : np.ndarray
        Extended LHZ grid.

    coord : tuple[int, int]
        Coordinate of the form ``(row, col)``.

    Returns
    -------
    tuple[int, int]
        Coordinate with the column index wrapped into the allowed range.
    """
    r, c = coord
    return (r, periodic_col(c, grid.shape[1]))


def canonical_state(state, grid: np.ndarray):
    """
    Canonicalize all coordinates in a spanning line by applying horizontal
    periodic boundary conditions.

    Parameters
    ----------
    state : tuple[tuple[int, int], ...]
        Coordinate representation of a spanning line.

    grid : np.ndarray
        Extended LHZ grid.

    Returns
    -------
    tuple
        Canonicalized coordinate spanning line.
    """
    return tuple(canonical_coord(grid, coord) for coord in state)


def labels_from_state(state, grid: np.ndarray):
    """
    Convert a coordinate spanning line into its corresponding parity-label
    spanning line.

    Parameters
    ----------
    state : tuple[tuple[int, int], ...]
        Coordinate representation of a spanning line.

    grid : np.ndarray
        Extended LHZ grid.

    Returns
    -------
    list[tuple]
        List of parity labels in top-to-bottom row order.
    """
    return [get_label(grid, coord) for coord in state]


def build_lhz_faces(grid: np.ndarray):
    """
    Build the local stabilizer faces of the extended LHZ grid.

    The returned faces include top boundary triangles, bottom boundary
    triangles, and bulk plaquettes/diamonds. Only faces whose parity labels
    have empty symmetric difference are retained.

    Parameters
    ----------
    grid : np.ndarray
        Extended LHZ grid.

    Returns
    -------
    list[dict]
        List of face dictionaries. Each dictionary contains:
            - ``"type"``: face type;
            - ``"coords"``: coordinates of the face nodes;
            - ``"labels"``: parity labels on those nodes.
    """
    n_rows, width = grid.shape
    faces = []
    seen = set()

    def add_face(face_type: str, coords):
        coords = [canonical_coord(grid, coord) for coord in coords]

        if not all(is_valid_node(grid, coord) for coord in coords):
            return

        labels = tuple(get_label(grid, coord) for coord in coords)

        if symdiff_labels(labels) != ():
            return

        key = frozenset(coords)
        if key in seen:
            return

        seen.add(key)

        faces.append(
            {
                "type": face_type,
                "coords": tuple(coords),
                "labels": labels,
            }
        )

    # Top boundary triangles:
    #   (0, c), (0, c+2), (1, c+1)
    for c in range(width):
        add_face(
            "top_triangle",
            [
                (0, c),
                (0, c + 2),
                (1, c + 1),
            ],
        )

    # Bottom boundary triangles:
    #   (n-1, c), (n-1, c+2), (n-2, c+1)
    for c in range(width):
        add_face(
            "bottom_triangle",
            [
                (n_rows - 1, c),
                (n_rows - 1, c + 2),
                (n_rows - 2, c + 1),
            ],
        )

    # Bulk plaquettes / diamonds:
    #   (r-1, c), (r, c-1), (r, c+1), (r+1, c)
    for r in range(1, n_rows - 1):
        for c in range(width):
            add_face(
                "plaquette",
                [
                    (r - 1, c),
                    (r, c - 1),
                    (r, c + 1),
                    (r + 1, c),
                ],
            )

    return faces


def coords_for_label_in_row(grid: np.ndarray, label, row: int):
    """
    Return all coordinates in a fixed row carrying a specified parity label.

    Parameters
    ----------
    grid : np.ndarray
        Extended LHZ grid.

    label : int, tuple, list, set, or frozenset
        Desired parity label.

    row : int
        Row index.

    Returns
    -------
    list[tuple[int, int]]
        Coordinates in the specified row with the requested label.
    """
    label = normalize_label(label)
    _, width = grid.shape

    coords = []
    for c in range(width):
        if grid[row, c] == label:
            coords.append((row, c))

    return coords


def is_valid_spanning_line_coords(
    coords,
    grid: np.ndarray,
    require_one_node_per_row: bool = True,
) -> bool:
    """
    Check whether a coordinate list defines a valid top-to-bottom spanning line.

    A valid spanning line contains one valid node per row and adjacent nodes in
    consecutive rows are connected by diagonal nearest-neighbour edges.

    Parameters
    ----------
    coords : list[tuple[int, int]]
        Candidate coordinate spanning line.

    grid : np.ndarray
        Extended LHZ grid.

    require_one_node_per_row : bool
        If True, require that the coordinates are ordered from row 0 to row
        ``n-1`` with exactly one node per row.

    Returns
    -------
    bool
        True if the coordinate list is a valid spanning line.
    """
    n_rows, width = grid.shape

    if len(coords) != n_rows:
        return False

    coords = [canonical_coord(grid, coord) for coord in coords]

    if require_one_node_per_row:
        rows = [r for r, _ in coords]
        if rows != list(range(n_rows)):
            return False

    for coord in coords:
        if not is_valid_node(grid, coord):
            return False

    for i in range(n_rows - 1):
        r1, c1 = coords[i]
        r2, c2 = coords[i + 1]

        if r2 != r1 + 1:
            return False

        if not periodic_distance_is_one(c1, c2, width):
            return False

    return True


def label_line_to_coordinate_lines(
    label_line,
    grid: np.ndarray,
    name: str = "spanning line",
):
    """
    Convert a parity-label spanning line into all valid coordinate realizations.

    Parameters
    ----------
    label_line : list
        Spanning line specified by parity labels in top-to-bottom row order.

    grid : np.ndarray
        Extended LHZ grid.

    name : str
        Name used in error messages.

    Returns
    -------
    list[tuple]
        List of valid coordinate realizations of the label spanning line.

    Raises
    ------
    ValueError
        If the label line cannot be embedded as a connected top-to-bottom
        spanning line.
    """
    n_rows, _ = grid.shape

    if len(label_line) != n_rows:
        raise ValueError(
            f"Invalid {name}: expected exactly {n_rows} labels, "
            f"got {len(label_line)}."
        )

    label_line = [normalize_label(label) for label in label_line]

    row_options = []
    for row, label in enumerate(label_line):
        options = coords_for_label_in_row(grid, label, row)

        if not options:
            raise ValueError(
                f"Invalid {name}: label {label} does not appear in row {row}. "
                "Therefore it cannot be part of a top-to-bottom spanning line "
                "in the specified row ordering."
            )

        row_options.append(options)

    valid_lines = []

    def backtrack(row: int, partial):
        if row == n_rows:
            if is_valid_spanning_line_coords(partial, grid):
                valid_lines.append(tuple(partial))
            return

        for coord in row_options[row]:
            if row > 0:
                _, prev_c = partial[-1]
                _, c = coord

                if not periodic_distance_is_one(prev_c, c, grid.shape[1]):
                    continue

            partial.append(coord)
            backtrack(row + 1, partial)
            partial.pop()

    backtrack(0, [])

    if not valid_lines:
        raise ValueError(
            f"Invalid {name}: labels {label_line} cannot be embedded as a "
            "connected top-to-bottom spanning line."
        )

    return valid_lines


def face_crossing_neighbors(state, grid: np.ndarray, faces):
    """
    Generate neighbouring spanning lines obtained by crossing one local LHZ face.

    A face crossing replaces exactly one node of the current spanning line by
    the missing node of a triangle or plaquette whose other nodes are already
    contained in the line. The corresponding CNOT metadata is also returned.

    Parameters
    ----------
    state : tuple
        Current coordinate spanning line.

    grid : np.ndarray
        Extended LHZ grid.

    faces : list[dict]
        Local stabilizer faces returned by ``build_lhz_faces``.

    Returns
    -------
    list[tuple]
        List of ``(new_state, cnot_steps)`` pairs. Each ``cnot_steps`` entry is
        a dictionary containing control/target physical qubits and label
        information for the corresponding CNOT update.
    """
    state_set = set(state)
    neighbors = []

    coord_to_row = {coord: row for row, coord in enumerate(state)}

    for face in faces:
        face_coords = set(face["coords"])

        contained = list(face_coords & state_set)
        missing = list(face_coords - state_set)

        if len(missing) != 1:
            continue

        if len(contained) != len(face_coords) - 1:
            continue

        missing_coord = missing[0]
        missing_row = missing_coord[0]

        possible_targets = [
            coord for coord in contained
            if coord[0] == missing_row and coord in coord_to_row
        ]

        if len(possible_targets) != 1:
            continue

        target_coord = possible_targets[0]
        target_index = coord_to_row[target_coord]

        control_coords = [
            coord for coord in contained
            if coord != target_coord
        ]

        target_label = get_label(grid, target_coord)
        control_labels = [get_label(grid, coord) for coord in control_coords]
        missing_label = get_label(grid, missing_coord)

        produced_label = symdiff_labels([target_label] + control_labels)

        if produced_label != missing_label:
            continue

        new_state = list(state)
        new_state[target_index] = missing_coord
        new_state = tuple(new_state)

        if not is_valid_spanning_line_coords(list(new_state), grid):
            continue

        cnot_steps = []
        for control_coord in control_coords:
            control_index = coord_to_row[control_coord]

            cnot_steps.append(
                {
                    "control": control_index,
                    "target": target_index,
                    "control_coord": control_coord,
                    "target_coord_before": target_coord,
                    "target_coord_after": missing_coord,
                    "control_label": get_label(grid, control_coord),
                    "target_label_before": target_label,
                    "target_label_after": missing_label,
                    "face_type": face["type"],
                    "face_coords": face["coords"],
                    "face_labels": face["labels"],
                }
            )

        neighbors.append((new_state, cnot_steps))

    return neighbors


def right_moving_face_crossings(state, grid: np.ndarray, faces):
    """
    Return all face crossings that move a spanning-line node to the right.

    A right-moving crossing is an elementary face crossing in which exactly one
    row changes and the new coordinate in that row is two columns to the right
    of the old coordinate, modulo the horizontal period.

    Parameters
    ----------
    state : tuple
        Current coordinate spanning line.

    grid : np.ndarray
        Extended LHZ grid.

    faces : list[dict]
        Local stabilizer faces returned by ``build_lhz_faces``.

    Returns
    -------
    list[dict]
        List of right-moving moves. Each move contains:
            - ``"row"``;
            - ``"old_coord"``;
            - ``"new_coord"``;
            - ``"new_state"``;
            - ``"cnot_steps"``;
            - face metadata.
    """
    width = grid.shape[1]
    moves = []

    neighbors = face_crossing_neighbors(state, grid, faces)

    for new_state, cnot_steps in neighbors:
        new_state = canonical_state(new_state, grid)

        changed_rows = [
            row
            for row, (old_coord, new_coord) in enumerate(zip(state, new_state))
            if canonical_coord(grid, old_coord) != canonical_coord(grid, new_coord)
        ]

        if len(changed_rows) != 1:
            continue

        row = changed_rows[0]

        old_coord = canonical_coord(grid, state[row])
        new_coord = canonical_coord(grid, new_state[row])

        old_r, old_c = old_coord
        new_r, new_c = new_coord

        if old_r != new_r:
            continue

        if periodic_col(old_c + 2, width) != new_c:
            continue

        moves.append(
            {
                "row": row,
                "old_coord": old_coord,
                "new_coord": new_coord,
                "new_state": new_state,
                "cnot_steps": cnot_steps,
                "face_type": cnot_steps[0]["face_type"] if cnot_steps else None,
                "face_coords": cnot_steps[0]["face_coords"] if cnot_steps else None,
                "face_labels": cnot_steps[0]["face_labels"] if cnot_steps else None,
            }
        )

    return moves


def cnot_steps_from_moves(moves):
    """
    Flatten CNOT metadata from a list of face moves.

    Parameters
    ----------
    moves : list[dict]
        Moves returned by ``right_moving_face_crossings`` or by a move-selection
        routine.

    Returns
    -------
    list[dict]
        Flattened list of CNOT-step metadata.
    """
    steps = []

    for move in moves:
        sorted_steps = sorted(
            move["cnot_steps"],
            key=lambda step: (step["target"], -step["control"]),
        )
        steps.extend(sorted_steps)

    return steps


def cnot_gate_list_from_steps(steps):
    """
    Convert CNOT-step metadata into gate-list format.

    Parameters
    ----------
    steps : list[dict]
        CNOT-step metadata.

    Returns
    -------
    list[list]
        CNOT gates in the format ``["CNOT", [control, target], "N/A"]``.
    """
    return [
        ["CNOT", [step["control"], step["target"]], "N/A"]
        for step in steps
    ]


def cnot_sequence_between_lhz_spanning_lines(
    start_line,
    final_line,
    n: int,
    max_search_steps=None,
    return_debug: bool = False,
):
    """
    Find a CNOT sequence between two valid LHZ spanning lines.

    The search uses local triangle/plaquette face crossings in the extended LHZ
    grid. Input and output spanning lines are specified by parity labels in
    top-to-bottom row order.

    Parameters
    ----------
    start_line : list
        Starting parity-label spanning line.

    final_line : list
        Target parity-label spanning line.

    n : int
        Number of logical qubits.

    max_search_steps : int or None
        Optional limit on the number of BFS expansions.

    return_debug : bool
        If True, return CNOT metadata, coordinates, grid, and faces.

    Returns
    -------
    list[tuple[int, int]] or dict
        If ``return_debug=False``, returns CNOTs as ``(control, target)``
        tuples. Otherwise returns a dictionary with debug information.
    """
    grid = build_extended_lhz_array(n)
    faces = build_lhz_faces(grid)

    start_coord_lines = label_line_to_coordinate_lines(
        start_line, grid, name="start_line"
    )

    final_coord_lines = label_line_to_coordinate_lines(
        final_line, grid, name="final_line"
    )

    final_states = set(final_coord_lines)

    queue = deque()
    visited = set()

    for start_state in start_coord_lines:
        queue.append((start_state, [], []))
        visited.add(start_state)

    expansions = 0

    while queue:
        state, cnot_sequence, cnot_steps = queue.popleft()

        if state in final_states:
            if return_debug:
                return {
                    "cnots": cnot_sequence,
                    "cnot_steps": cnot_steps,
                    "start_coords": state if len(cnot_steps) == 0 else None,
                    "final_coords": state,
                    "grid": grid,
                    "faces": faces,
                }

            return cnot_sequence

        if max_search_steps is not None and expansions >= max_search_steps:
            break

        expansions += 1

        for new_state, steps in face_crossing_neighbors(state, grid, faces):
            if new_state in visited:
                continue

            visited.add(new_state)

            new_cnot_sequence = list(cnot_sequence)
            for step in steps:
                new_cnot_sequence.append((step["control"], step["target"]))

            new_cnot_steps = cnot_steps + steps

            queue.append((new_state, new_cnot_sequence, new_cnot_steps))

    raise ValueError(
        "No path found between the two spanning lines using only geometric "
        "triangle/plaquette moves in the extended LHZ scheme."
    )