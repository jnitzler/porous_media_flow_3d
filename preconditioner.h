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

#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

namespace Preconditioner
{
  using namespace dealii;

  // ===========================================================================
  // InverseMatrix: Wrapper to apply M^{-1} via iterative CG solve.
  // Used as a building block in the Schur complement preconditioner.
  // ===========================================================================
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public EnableObserverPointer
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
    // Use fixed iteration count instead of tight tolerance to limit
    // per-application cost while maintaining preconditioner quality
    IterationNumberControl solver_control(5, 1e-14);
    SolverCG<VectorType>   cg(solver_control);

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
  // vmult solves the lower-triangular system
  //   [ M   0  ] [ u ]   [ f ]
  //   [ B   -S ] [ p ] = [ g ]
  //
  // The Darcy saddle-point matrix is symmetric, so the same preconditioner
  // serves the adjoint solve as well — no Tvmult is required.
  // ===========================================================================
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  class BlockSchurPreconditioner : public EnableObserverPointer
  {
  public:
    BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &System,
                             const PreconditionerTypeaS &ap_S_inv,
                             const PreconditionerTypeM  &ap_M_inv,
                             TimerOutput                &computing_timer);

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const;

  private:
    const TrilinosWrappers::BlockSparseMatrix &system_matrix;
    const PreconditionerTypeaS                &ap_S_inv; // Approximate S^{-1}
    const PreconditionerTypeM                 &ap_M_inv; // Approximate M^{-1}
    TimerOutput                               &computing_timer;
    mutable TrilinosWrappers::MPI::Vector tmp; // Temporary (pressure-sized)
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

} // namespace Preconditioner

#endif // PRECONDITIONER_H
