// preconditioner.h
// Block Schur complement preconditioner for the Darcy saddle-point system.
//
// System structure:
//   [ M    B^T ] [ u ]   [ f ]
//   [ B    0   ] [ p ] = [ g ]
//
// Preconditioner approximates the block LU factorization using:
//   - Incomplete Cholesky on M (velocity mass-like block)
//   - Incomplete Cholesky on approximate Schur complement S ≈ B M^{-1} B^T

#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "darcy.h"

namespace Preconditioner
{
  using namespace dealii;

  // ===========================================================================
  // InverseMatrix: Wrapper to apply M^{-1} via iterative CG solve.
  // Used as a building block in the Schur complement preconditioner.
  // ===========================================================================
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType         &m,
                  const PreconditionerType &preconditioner);

    // Apply dst = M^{-1} * src using preconditioned CG
    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    const MatrixType         &matrix;
    const PreconditionerType &preconditioner;
  };

  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType         &m,
    const PreconditionerType &preconditioner)
    : matrix(m)
    , preconditioner(preconditioner)
  {}

  template <class MatrixType, class PreconditionerType>
  template <typename VectorType>
  void
  InverseMatrix<MatrixType, PreconditionerType>::vmult(
    VectorType       &dst,
    const VectorType &src) const
  {
    // Use ReductionControl for both absolute and relative tolerance
    // This ensures consistent convergence regardless of src magnitude
    const double         abs_tol = 1e-14;
    const double         rel_tol = 1e-10;
    ReductionControl     solver_control(src.size(), abs_tol, rel_tol);
    SolverCG<VectorType> cg(solver_control);

    dst = 0;

    try
      {
        cg.solve(matrix, dst, src, preconditioner);
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }
  }

  // ===========================================================================
  // BlockSchurPreconditioner: Block triangular preconditioner for saddle-point.
  //
  // Forward (vmult): Solves lower triangular system
  //   [ M   0  ] [ u ]   [ f ]
  //   [ B   -S ] [ p ] = [ g ]
  //
  // Transpose (Tvmult): Solves upper triangular system (for adjoint)
  //   [ M   B^T ] [ u ]   [ f ]
  //   [ 0   -S  ] [ p ] = [ g ]
  // ===========================================================================
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &System,
                             const PreconditionerTypeaS &ap_S_inv,
                             const PreconditionerTypeM  &ap_M_inv,
                             TimerOutput                &computing_timer);

    // Forward solve (lower triangular block)
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const;

    // Transpose solve (upper triangular block) - used for adjoint
    void
    Tvmult(TrilinosWrappers::MPI::BlockVector       &dst,
           const TrilinosWrappers::MPI::BlockVector &src) const;

    // Matrix dimensions (required for TransposeOperator wrapper)
    types::global_dof_index
    m() const
    {
      return system_matrix.m();
    }

    types::global_dof_index
    n() const
    {
      return system_matrix.n();
    }

  private:
    const TrilinosWrappers::BlockSparseMatrix &system_matrix;
    const PreconditionerTypeaS                &ap_S_inv; // Approximate S^{-1}
    const PreconditionerTypeM                 &ap_M_inv; // Approximate M^{-1}
    TimerOutput                               &computing_timer;
    mutable TrilinosWrappers::MPI::Vector tmp;   // Temporary (pressure-sized)
    mutable TrilinosWrappers::MPI::Vector tmp_u; // Temporary (velocity-sized)
  };

  // ===========================================================================
  // Implementation
  // ===========================================================================

  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::
    BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &System,
                             const PreconditionerTypeaS &ap_S_inv,
                             const PreconditionerTypeM  &ap_M_inv,
                             TimerOutput                &computing_timer)
    : system_matrix(System)
    , ap_S_inv(ap_S_inv)
    , ap_M_inv(ap_M_inv)
    , computing_timer(computing_timer)
    , tmp(System.block(1, 1).locally_owned_range_indices(),
          System.get_mpi_communicator())
    , tmp_u(System.block(0, 0).locally_owned_range_indices(),
            System.get_mpi_communicator())
  {}

  // Forward solve: [ M 0; B -S ] * [u; p] = [f; g]
  // Step 1: u = M^{-1} f
  // Step 2: p = S^{-1} (B u - g)
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  void
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::vmult(
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Apply preconditioner");
    // Step 1: Solve M * u = f
    ap_M_inv.vmult(dst.block(0), src.block(0));

    // Step 2: Compute residual r_p = g - B * u
    system_matrix.block(1, 0).residual(tmp, dst.block(0), src.block(1));

    // Step 3: Solve -S * p = r_p  =>  p = -S^{-1} * r_p
    tmp *= -1;
    ap_S_inv.vmult(dst.block(1), tmp);
  }

  // Transpose solve: [ M B^T; 0 -S ] * [u; p] = [f; g]
  // Step 1: p = -S^{-1} g
  // Step 2: u = M^{-1} (f - B^T p)
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  void
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::Tvmult(
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Apply transpose preconditioner");

    // Step 1: Solve -S * p = g  =>  p = -S^{-1} * g
    ap_S_inv.vmult(dst.block(1), src.block(1));
    dst.block(1) *= -1.0;

    // Step 2: Compute B^T * p (using Tvmult on B block)
    system_matrix.block(1, 0).Tvmult(tmp_u, dst.block(1));

    // Step 3: Compute RHS for velocity: tmp_u = f - B^T * p
    tmp_u.sadd(-1.0, 1.0, src.block(0));

    // Step 4: Solve M * u = tmp_u
    ap_M_inv.vmult(dst.block(0), tmp_u);
  }

} // namespace Preconditioner

#endif // PRECONDITIONER_H
