// preconditioner.h
#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "darcy.h"

namespace Preconditioner
{
  using namespace dealii;

  // -----  inverse matrix class ----------------------
  template <class MatrixType, class PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const MatrixType         &m,
                  const PreconditionerType &preconditioner);

    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    const MatrixType         &matrix;
    const PreconditionerType &preconditioner;
  };

  // constructor
  template <class MatrixType, class PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType         &m,
    const PreconditionerType &preconditioner)
    : matrix(m)
    , preconditioner(preconditioner)
  {}

  // vmult function
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

  // ---------------------------------------------------------------------------
  // Block Schur Preconditioner
  // ---------------------------------------------------------------------------
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &System,
                             const PreconditionerTypeaS &ap_S_inv,
                             const PreconditionerTypeM  &ap_M_inv,
                             TimerOutput                &computing_timer);

    // Standard Forward Solve (Lower Triangular Block)
    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const;

    // Transpose Solve (Upper Triangular Block)
    void
    Tvmult(TrilinosWrappers::MPI::BlockVector       &dst,
           const TrilinosWrappers::MPI::BlockVector &src) const;

    // Dimensions (Delegated to system matrix)
    // Required for the TransposeOperator wrapper to work
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
    const PreconditionerTypeaS                &ap_S_inv;
    const PreconditionerTypeM                 &ap_M_inv;
    TimerOutput                               &computing_timer;
    mutable TrilinosWrappers::MPI::Vector      tmp; // size n_p (pressure block)
    mutable TrilinosWrappers::MPI::Vector tmp_u;    // size n_u (velocity block)
  };

  // ---------------------------------------------------------------------------
  // Implementation
  // ---------------------------------------------------------------------------

  // Constructor
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

  // vmult: Solves Lower Triangular System
  // [ M   0  ] [ u ]   [ f ]
  // [ B   -S ] [ p ] = [ g ]
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  void
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::vmult(
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Apply preconditioner");
    // 1. Solve M * u = f
    ap_M_inv.vmult(dst.block(0), src.block(0));

    // 2. Update RHS for p: r_p = g - B * u
    system_matrix.block(1, 0).residual(tmp, dst.block(0), src.block(1));

    // 3. Solve -S * p = r_p
    tmp *= -1;
    ap_S_inv.vmult(dst.block(1), tmp);
  }

  // Tvmult: Solves Upper Triangular System (Transpose of vmult)
  //
  // The forward preconditioner approximates:
  // [ M       0 ] [ u ]   [ f ]
  // [ B_10   -S ] [ p ] = [ g ]
  //
  // The transpose of this lower triangular system is upper triangular:
  // [ M^T    B_10^T ] [ u ]   [ f ]
  // [ 0       -S^T  ] [ p ] = [ g ]
  //
  // Since M and S are symmetric (M^T = M, S^T = S), and B_10^T means
  // we need to apply the TRANSPOSE of block(1,0):
  //
  // [ M      B_10^T ] [ u ]   [ f ]
  // [ 0       -S    ] [ p ] = [ g ]
  //
  // Solving this upper triangular system:
  // 1. From row 2: -S * p = g  =>  p = -S_inv * g
  // 2. From row 1: M * u = f - B_10^T * p  =>  u = M_inv * (f - B_10^T * p)
  //
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  void
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::Tvmult(
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Apply transpose preconditioner");

    // 1. Solve for p (Second block): -S * p = g
    ap_S_inv.vmult(dst.block(1), src.block(1));
    dst.block(1) *= -1.0;

    // 2. Update RHS for u: M * u = f - B_10^T * p
    // Calculate B_10^T * p into tmp_u (velocity-sized vector)
    // B_10 = system_matrix.block(1, 0) has dims n_p x n_u
    // B_10^T has dims n_u x n_p, so B_10^T * p gives velocity-sized vector
    // We use Tvmult on block(1,0) to compute B_10^T * p
    system_matrix.block(1, 0).Tvmult(tmp_u, dst.block(1));

    // Calculate effective RHS: tmp_u = f - (B_10^T * p) = src.block(0) - tmp_u
    tmp_u.sadd(-1.0, 1.0, src.block(0));

    // 3. Solve for u: u = M_inv * tmp_u
    ap_M_inv.vmult(dst.block(0), tmp_u);
  }
} // end namespace Preconditioner
#endif
