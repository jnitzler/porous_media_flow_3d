#!/usr/bin/env python3
"""Compute initial variational parameters from the prior precision matrix.

Loads the lower-triangular COO sparsity pattern and matrix values exported by
``export_sparsity``, computes a sparse Cholesky factorization via CHOLMOD
(truncated to the sparsity pattern), and writes the result in QUEENS format.

Modes:
    sparse_inverse (default):
        Scales IC(0) to target marginal variance and writes parameters in
        SparseNormalInverseVariational format (precision Cholesky L_Q).

    whitened:
        Exports unscaled IC(0) as L_0 (prior precision Cholesky) and writes
        trivial initial variational parameters (m=0, L_S=I) for the
        WhitenedSparseNormalVariational distribution.

Usage:
    python compute_prior_init.py [--mode sparse_inverse] [--target-variance 1.0] [--mean 0.1]
    python compute_prior_init.py --mode whitened [--mean 0.1]

Inputs (in CWD):
    rf_sparsity_row_idx.npy
    rf_sparsity_col_idx.npy
    rf_A_kappa_values.npy

Output (sparse_inverse mode):
    initial_variational_params.npy

Output (whitened mode):
    prior_cholesky_L0.npy           — IC(0) L_0 values (same COO order as sparsity)
    initial_variational_params.npy  — all zeros (m=0, L_S=I)
"""

import argparse

import numpy as np
from scipy import sparse
from sksparse.cholmod import cholesky


def load_A_kappa(row_file, col_file, val_file):
    """Load lower-triangular COO data and build the full symmetric A_kappa."""
    rows = np.load(row_file).flatten().astype(np.int64)
    cols = np.load(col_file).flatten().astype(np.int64)
    vals = np.load(val_file).flatten()

    n = int(max(rows.max(), cols.max())) + 1

    # Build lower triangle
    A_lower = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))

    # Full symmetric: A = L + L^T - diag(L)
    A = A_lower + A_lower.T - sparse.diags(A_lower.diagonal())

    return A, rows, cols


def sparse_cholesky(A, rows, cols):
    """Compute sparse Cholesky via CHOLMOD and extract at prescribed sparsity.

    CHOLMOD computes the exact sparse Cholesky L_full of A (with fill-in).
    We then extract L values at the original sparsity positions (rows, cols),
    effectively truncating fill-in — similar to IC(0) but much faster.

    Args:
        A: Full symmetric SPD matrix (scipy sparse).
        rows: Lower-triangular row indices of target sparsity pattern.
        cols: Lower-triangular column indices of target sparsity pattern.

    Returns:
        L_sparse: Sparse lower-triangular matrix at the target sparsity pattern.
        L_vals: Values array matching (rows, cols) ordering.
    """
    # CHOLMOD requires CSC format
    A_csc = sparse.csc_matrix(A)
    factor = cholesky(A_csc)
    L_full = factor.L()
    P = factor.P()  # permutation: P A P^T = L L^T

    print(f"  CHOLMOD L: {L_full.nnz} nnz (full), extracting {len(rows)} at sparsity pattern")

    # Unpermute: L_original = P^T L_full P (apply inverse permutation)
    # P[i] = j means original row i maps to permuted row j
    inv_P = np.empty_like(P)
    inv_P[P] = np.arange(len(P))
    L_unperm = L_full[np.ix_(inv_P, inv_P)]

    # Extract values at the prescribed sparsity pattern
    L_csc = L_unperm.tocsc()
    L_vals = np.array(L_csc[rows, cols]).flatten()

    # Build sparse matrix at target sparsity
    L_sparse = sparse.csr_matrix(
        sparse.coo_matrix((L_vals, (rows, cols)), shape=(A.shape[0], A.shape[0]))
    )

    return L_sparse, L_vals


def scale_L_values(L_vals, rows, cols, n, target_variance):
    """Scale L_Q values so average marginal variance matches target.

    For precision Q = L_Q @ L_Q^T, the marginal variance at DOF i is
    var[i] = (Q^{-1})_{ii} ≈ 1 / Q_{ii} = 1 / sum_j L_Q[i,j]^2.

    We scale L_Q so that mean(1 / sum_j L_Q[i,j]^2) ≈ target_variance.
    """
    # Compute row-wise sum of squares
    row_sq_sums = np.zeros(n)
    for k in range(len(rows)):
        row_sq_sums[rows[k]] += L_vals[k] ** 2

    avg_marginal_var = np.mean(1.0 / row_sq_sums)
    print(f"  Before scaling: avg marginal variance = {avg_marginal_var:.6e}")

    # scale^2 * row_sq_sums -> want mean(1/(scale^2 * row_sq_sums)) = target
    scale = 1.0 / np.sqrt(target_variance * np.mean(row_sq_sums))
    L_vals_scaled = L_vals * scale

    scaled_row_sq = row_sq_sums * scale**2
    avg_var_after = np.mean(1.0 / scaled_row_sq)
    print(f"  After scaling:  avg marginal variance = {avg_var_after:.6e}")
    print(f"  Scale factor: {scale:.6e}")

    return L_vals_scaled


def to_queens_format(mean_vector, L_values, rows, cols):
    """Convert to QUEENS SparseNormalInverseVariational parameter format.

    Format: [mu_0, ..., mu_{d-1}, lambda_0, ..., lambda_{nnz-1}]
    where lambda_i = inverse_softplus(L_Q[i,i]) for diagonal (softplus transform)
          lambda_i = L_Q[i,j]                   for off-diagonal (identity)
    """
    is_diag = rows == cols
    chol_params = L_values.copy()

    diag_vals = chol_params[is_diag]
    if np.any(diag_vals <= 0):
        raise ValueError(
            f"Found {np.sum(diag_vals <= 0)} non-positive diagonal entries in L_Q."
        )

    # Inverse softplus: λ = log(exp(y) - 1), numerically stable for large y
    chol_params[is_diag] = np.where(
        diag_vals > 20, diag_vals, np.log(np.expm1(diag_vals))
    )

    print(f"  Diagonal L_Q range: [{diag_vals.min():.4e}, {diag_vals.max():.4e}]")
    print(
        f"  Off-diagonal L_Q range: [{chol_params[~is_diag].min():.4e}, "
        f"{chol_params[~is_diag].max():.4e}]"
    )

    return np.concatenate([mean_vector, chol_params])


def main():
    parser = argparse.ArgumentParser(
        description="Compute initial variational parameters from prior precision."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sparse_inverse", "whitened"],
        default="sparse_inverse",
        help="Output mode: 'sparse_inverse' (precision L_Q, default) or "
             "'whitened' (exports L_0 + trivial init for whitened distribution)",
    )
    parser.add_argument(
        "--target-variance",
        type=float,
        default=1.0,
        help="Target average marginal variance (sparse_inverse mode only, default: 1.0)",
    )
    parser.add_argument(
        "--mean",
        type=float,
        default=0.1,
        help="Initial mean value for all DOFs (default: 0.1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="initial_variational_params.npy",
        help="Output file path for variational params (default: initial_variational_params.npy)",
    )
    args = parser.parse_args()

    print("Loading COO sparsity and A_kappa values...")
    A_kappa, rows, cols = load_A_kappa(
        "rf_sparsity_row_idx.npy",
        "rf_sparsity_col_idx.npy",
        "rf_A_kappa_values.npy",
    )
    n = A_kappa.shape[0]
    nnz = len(rows)
    print(f"  A_kappa: {n} x {n}, lower-tri nnz = {nnz}")
    print(f"  A_kappa diagonal range: [{A_kappa.diagonal().min():.4e}, "
          f"{A_kappa.diagonal().max():.4e}]")

    print("Computing sparse Cholesky via CHOLMOD...")
    L, L_vals = sparse_cholesky(A_kappa, rows, cols)
    print(f"  L nnz (truncated) = {L.nnz}")

    if args.mode == "whitened":
        # Export unscaled L_0 values for WhitenedSparseNormalVariational
        l0_file = "prior_cholesky_L0.npy"
        print(f"Saving prior Cholesky L_0 values to {l0_file}...")
        np.save(l0_file, L_vals)

        # Compute approximate prior variance to scale L_S for target variance.
        # Physical variance: var_i ≈ s² / sum_j L_0[i,j]², where s = L_S diagonal.
        is_diag_sp = rows == cols
        row_sq_sums = np.zeros(n)
        for k in range(len(rows)):
            row_sq_sums[rows[k]] += L_vals[k] ** 2
        avg_prior_var = np.mean(1.0 / row_sq_sums)
        s = np.sqrt(args.target_variance / avg_prior_var)
        log_s = np.log(s)
        print(f"  Approximate avg prior variance: {avg_prior_var:.4f}")
        print(f"  Scaling L_S diagonal to s={s:.6f} (log(s)={log_s:.4f}) "
              f"for target variance {args.target_variance}")

        # Init: m=0 (whitened mean), L_S=s*I (scaled for target variance)
        # Diagonal λ=log(s), off-diagonal λ=1e-2 → (1e-2)^3=1e-6 ≈ 0
        # Small non-zero off-diag avoids dead zone in cubic transform gradient (3λ²=3e-4)
        variational_params = np.zeros(n + nnz)
        variational_params[n:][is_diag_sp] = log_s
        variational_params[n:][~is_diag_sp] = 1e-2
        print(f"Saving init (m=0, L_S=s*I, off-diag λ=1e-2) to {args.output}...")
        np.save(args.output, variational_params)

        diag_vals = L_vals[is_diag_sp]
        print(f"\nSummary (whitened mode):")
        print(f"  Dimension: {n}")
        print(f"  Cholesky nnz (truncated): {nnz}")
        print(f"  L_0 diagonal range: [{diag_vals.min():.4e}, {diag_vals.max():.4e}]")
        print(f"  L_S init diagonal: exp({log_s:.4f}) = {s:.6f}")
        print(f"  Physical-space avg variance: ~{args.target_variance}")
        print(f"  Prior mean: {args.mean} (pass as prior_mean in YAML)")
        print(f"  Total variational parameters: {n + nnz} = {n} (m) + {nnz} (L_S)")
        print(f"  Output files: {l0_file}, {args.output}")
    else:
        print(f"Scaling to target variance = {args.target_variance}...")
        L_vals_scaled = scale_L_values(L_vals, rows, cols, n, args.target_variance)

        print("Converting to QUEENS format...")
        mean_vector = args.mean * np.ones(n)
        variational_params = to_queens_format(mean_vector, L_vals_scaled, rows, cols)

        print(f"Saving to {args.output}...")
        np.save(args.output, variational_params)

        n_diag = np.sum(rows == cols)
        n_offdiag = nnz - n_diag
        print(f"\nSummary (sparse_inverse mode):")
        print(f"  Dimension: {n}")
        print(f"  Cholesky nnz (truncated): {nnz} ({n_diag} diagonal + {n_offdiag} off-diagonal)")
        print(f"  Total parameters: {len(variational_params)} = {n} (mean) + {nnz} (L_Q)")
        print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
