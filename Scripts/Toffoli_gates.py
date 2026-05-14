import itertools
import numpy as np


def generate_cnx_logical_z_gates(n_controls, controls=None):
    """
    Generate the logical phase-gadget decomposition of an n-controlled X gate.

    The decomposition is based on

        C^n X = H_t C^n Z H_t,

    and

        C^n Z ~ prod_{empty != S subset Q}
            exp[
                i pi / 2^(n+1) (-1)^|S| prod_{q in S} Z_q
            ],

    where Q contains the n controls and the target.

    Using the convention

        RZ_S(theta) = exp[-i theta/2 prod_{q in S} Z_q],

    the angle for each subset S is

        theta_S = (-1)^(|S|+1) pi / 2^n.

    Parameters
    ----------
    n_controls : int
        Number of control qubits.

    controls : list[int] or None
        Control qubit labels. If None, uses [0, 1, ..., n_controls-1].

    Wire convention
    ---------------
    The target qubit is fixed to

        target = n_controls.

    The clean ancilla used later for decomposing many-body RZ rotations is
    reserved as

        ancilla = n_controls + 1.

    The ancilla is not part of this initial logical C^nZ decomposition.

    Returns
    -------
    gates : list[list]
        Gate list in the format used throughout the project:

            ["H", [target], "N/A"]
            ["RZ", [q1, q2, ...], theta]
            ["H", [target], "N/A"]

        The RZ gates include all non-empty subsets of controls + target.
    """
    if n_controls < 1:
        raise ValueError("n_controls must be at least 1.")

    if controls is None:
        controls = list(range(n_controls))

    if len(controls) != n_controls:
        raise ValueError(
            f"Expected {n_controls} controls, got {len(controls)}."
        )

    target = n_controls
    ancilla = n_controls + 1

    if target in controls:
        raise ValueError(
            "The fixed target qubit n_controls cannot also be a control qubit."
        )

    if ancilla in controls:
        raise ValueError(
            "The reserved ancilla qubit n_controls + 1 cannot be a control qubit."
        )

    qubits = list(controls) + [target]

    gates = []

    # First Hadamard on the target.
    gates.append(["H", [target], "N/A"])

    # Logical Z rotations for all non-empty subsets S of Q.
    for subset_size in range(1, len(qubits) + 1):
        for subset in itertools.combinations(qubits, subset_size):
            angle = ((-1) ** (subset_size + 1)) * np.pi / (2 ** n_controls)
            gates.append(["RZ", list(subset), angle])

    # Final Hadamard on the target.
    gates.append(["H", [target], "N/A"])
 
    return gates


def decompose_many_body_rz_to_two_body(gates, ancilla):
    """
    Decompose only genuinely many-body logical RZ rotations into CNOT cascades
    and a single-body RZ rotation on a clean ancilla.

    The input gate list is assumed to contain logical RZ rotations of the form

        ["RZ", [q_1, ..., q_m], theta]

    representing

        exp[-i theta/2 Z_{q_1} ... Z_{q_m}].

    Single-body and two-body RZ rotations are left unchanged. For every RZ gate
    acting on three or more qubits, this function implements

        R_{Z_S}(theta)
        =
        [prod_{j=1}^{m} CNOT(q_j, a)]
        R_{Z_a}(theta)
        [prod_{j=m}^{1} CNOT(q_j, a)],

    where S = {q_1, ..., q_m} and a is a clean ancilla initialized in |0>.

    The first CNOT cascade computes the full parity of S into the ancilla, the
    single-qubit RZ(theta) applies the corresponding parity phase, and the
    inverse CNOT cascade uncomputes the ancilla.

    Parameters
    ----------
    gates : list
        List of gates in the format [gate_name, qubits, parameter].
        For example:
            ["RZ", [0, 1, 2], theta]
            ["RZ", [0, 1], theta]
            ["CNOT", [0, 1], "N/A"]

    ancilla : int
        Index of the clean ancilla qubit used to store the parity.

    Returns
    -------
    new_gates : list
        Gate list where every RZ rotation acting on three or more qubits has
        been replaced by CNOT cascades and a single-qubit RZ rotation on the
        ancilla. One- and two-body RZ rotations are left unchanged.
    """

    new_gates = []

    for gate in gates:
        gate_name, qubits, param = gate

        # Only decompose logical RZ rotations.
        if gate_name != "RZ":
            new_gates.append(gate)
            continue

        # Keep single-body and two-body RZ rotations unchanged.
        if len(qubits) <= 2:
            new_gates.append(gate)
            continue

        # For |S| > 2, compute the full parity into the ancilla.
        for q in qubits:
            new_gates.append(["CNOT", [q, ancilla], "N/A"])

        # Apply single-body RZ rotation on the ancilla.
        new_gates.append(["RZ", [ancilla], param])

        # Uncompute the parity.
        for q in reversed(qubits):
            new_gates.append(["CNOT", [q, ancilla], "N/A"])

    return new_gates

def compress_cnx_decomposed_gates(gates, atol=0, max_passes=10_000):
    """
    Compress a CnX decomposition containing H gates, CNOTs, and one- or
    two-body RZ rotations.

    Important:
    The initial block before the first CNOT is kept fixed. In the CnX
    decomposition this block contains the first Hadamard and the initial
    one- and two-body logical Z rotations. These gates are not commuted with
    later gates.

    Compression rules applied after the initial fixed block:
      1. Identical adjacent CNOTs cancel.
      2. RZ rotations with the same support combine.
      3. CNOT-CNOT commutation is used only when valid.
      4. RZ-CNOT commutation is used only when valid.
      5. H gates are treated as barriers.
    """

    def normalize_gate(gate):
        gate_type, wires, angle = gate
        gate_type = gate_type.upper()

        if gate_type == "RZ":
            wires = sorted(wires)

            if len(wires) not in (1, 2):
                raise ValueError(
                    f"compress_cnx_decomposed_gates expects only one- or "
                    f"two-body RZ rotations, got {gate}."
                )

            return ["RZ", wires, angle]

        if gate_type == "CNOT":
            if len(wires) != 2:
                raise ValueError(f"CNOT must act on two wires, got {gate}.")
            return ["CNOT", list(wires), "N/A"]

        if gate_type == "H":
            if len(wires) != 1:
                raise ValueError(f"H must act on one wire, got {gate}.")
            return ["H", list(wires), "N/A"]

        raise ValueError(
            f"Unsupported gate type {gate_type}. Expected H, CNOT, or RZ."
        )

    def is_zero_rz(gate):
        gate_type, _, angle = gate
        return gate_type == "RZ" and abs(angle) <= atol

    def same_cnot(g1, g2):
        return (
            g1[0] == "CNOT"
            and g2[0] == "CNOT"
            and g1[1] == g2[1]
        )

    def same_rz_support(g1, g2):
        return (
            g1[0] == "RZ"
            and g2[0] == "RZ"
            and tuple(g1[1]) == tuple(g2[1])
        )

    def cnot_cnot_commute(g1, g2):
        c1, t1 = g1[1]
        c2, t2 = g2[1]

        return c1 != t2 and t1 != c2

    def rz_cnot_commute(rz_gate, cnot_gate):
        rz_support = set(rz_gate[1])
        _, target = cnot_gate[1]

        return target not in rz_support

    def gates_commute(g1, g2):
        """
        Conservative commutation checker.

        H gates are barriers. The initial RZ block is not passed to this
        function, so it is automatically protected from commutation.
        """
        if g1[0] == "H" or g2[0] == "H":
            return False

        if g1[0] == "RZ" and g2[0] == "RZ":
            return True

        if g1[0] == "CNOT" and g2[0] == "CNOT":
            return cnot_cnot_commute(g1, g2)

        if g1[0] == "RZ" and g2[0] == "CNOT":
            return rz_cnot_commute(g1, g2)

        if g1[0] == "CNOT" and g2[0] == "RZ":
            return rz_cnot_commute(g2, g1)

        return False

    def gate_sort_key(gate):
        """
        Canonical ordering key used only when two adjacent gates commute.
        """
        gate_type, wires, _ = gate

        if gate_type == "CNOT":
            c, t = wires
            return (0, c, t)

        if gate_type == "RZ":
            return (1, tuple(wires))

        if gate_type == "H":
            return (2, tuple(wires))

        return (99, tuple(wires))

    def compress_block(ops):
        """
        Compress a block of operations without moving through H barriers.
        """
        ops = [gate for gate in ops if not is_zero_rz(gate)]

        changed = True
        passes = 0

        while changed:
            passes += 1

            if passes > max_passes:
                raise RuntimeError(
                    "Maximum number of compression passes exceeded. "
                    "This likely indicates an unexpected rewrite loop."
                )

            changed = False
            new_ops = []
            i = 0

            while i < len(ops):
                if is_zero_rz(ops[i]):
                    changed = True
                    i += 1
                    continue

                if i + 1 < len(ops) and same_cnot(ops[i], ops[i + 1]):
                    changed = True
                    i += 2
                    continue

                if i + 1 < len(ops) and same_rz_support(ops[i], ops[i + 1]):
                    angle = ops[i][2] + ops[i + 1][2]

                    if abs(angle) > atol:
                        new_ops.append(["RZ", list(ops[i][1]), angle])

                    changed = True
                    i += 2
                    continue

                new_ops.append(ops[i])
                i += 1

            ops = new_ops

            i = 0
            while i < len(ops) - 1:
                g1 = ops[i]
                g2 = ops[i + 1]

                if gates_commute(g1, g2) and gate_sort_key(g1) > gate_sort_key(g2):
                    ops[i], ops[i + 1] = ops[i + 1], ops[i]
                    changed = True

                    if i > 0:
                        i -= 1
                    else:
                        i += 1
                else:
                    i += 1

        return ops

    # Normalize input.
    ops = [normalize_gate(gate) for gate in gates]

    # Protect the initial block before the first CNOT.
    #
    # This block contains:
    #   H_t,
    #   all initial one-body RZ rotations,
    #   all initial two-body RZ rotations.
    #
    # For n controls this is the block of size
    #   1 + (n+1) + binom(n+1, 2),
    # because the C^nZ acts on n controls plus one target.
    first_cnot_index = None

    for idx, gate in enumerate(ops):
        if gate[0] == "CNOT":
            first_cnot_index = idx
            break

    if first_cnot_index is None:
        # No CNOTs: only remove zero-angle RZs outside H barriers.
        return [gate for gate in ops if not is_zero_rz(gate)]

    fixed_initial_block = ops[:first_cnot_index]
    compressible_block = ops[first_cnot_index:]

    # fixed_initial_block = ops[:1]
    # compressible_block = ops[1:]

    compressed_block = compress_block(compressible_block)

    return fixed_initial_block + compressed_block