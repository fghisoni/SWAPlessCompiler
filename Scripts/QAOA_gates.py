import numpy as np


def generate_qaoa_ux_gates(n, beta):
    """
    Generate the gate list for the QAOA mixer unitary

        U_X(beta) = exp(i beta H_X),

    with

        H_X = sum_j X_j.

    Since all X_j terms commute, this is implemented as one RX-type logical
    rotation per qubit.

    Parameters
    ----------
    n : int
        Number of logical qubits.

    beta : float
        QAOA mixer angle.

    Returns
    -------
    gates : list[list]
        Gate list in the format

            ["RX", [j], beta]

        for j = 0, ..., n-1.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    gates = []

    for j in range(n):
        gates.append(["RX", [j], beta])

    return gates

def generate_qaoa_up_gates(n, gamma, J=None, h=None):
    """
    Generate the gate list for the QAOA problem unitary

        U_P(gamma) = exp(i gamma H_P),

    with

        H_P = sum_{k=1}^n sum_{j<k} J_{jk} Z_j Z_k
              + sum_j h_j Z_j.

    The output is already expressed in terms of logical RZ parity rotations:

        exp(i gamma J_jk Z_j Z_k) -> ["RZ", [j, k], gamma * J_jk]
        exp(i gamma h_j Z_j)      -> ["RZ", [j], gamma * h_j]

    Parameters
    ----------
    n : int
        Number of logical qubits.

    gamma : float
        QAOA problem angle.

    J : array-like, dict, or None
        Coupling coefficients J_jk.

        Supported formats:
            - None:
                all J_jk are set to 1.
            - numpy array or nested list of shape (n, n):
                J[j, k] is used for j < k.
            - dict:
                keys should be pairs (j, k), with j < k or k < j.

    h : array-like, dict, or None
        Local-field coefficients h_j.

        Supported formats:
            - None:
                all h_j are set to 0.
            - list or numpy array of length n:
                h[j] is used.
            - dict:
                h[j] is used for keys present in the dictionary.

    Returns
    -------
    gates : list[list]
        Gate list in the format

            ["RZ", [j, k], gamma * J_jk]
            ["RZ", [j], gamma * h_j]
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    gates = []

    def get_J(j, k):
        if J is None:
            return 1.0

        if isinstance(J, dict):
            if (j, k) in J:
                return J[(j, k)]
            if (k, j) in J:
                return J[(k, j)]
            return 0.0

        return J[j][k]

    def get_h(j):
        if h is None:
            return 0.0

        if isinstance(h, dict):
            return h.get(j, 0.0)

        return h[j]

    # Two-body ZZ terms.
    for k in range(n):
        for j in range(k):
            coeff = get_J(j, k)

            if coeff != 0:
                gates.append(["RZ", [j, k], gamma * coeff])

    # One-body Z terms.
    for j in range(n):
        coeff = get_h(j)

        if coeff != 0:
            gates.append(["RZ", [j], gamma * coeff])

    return gates