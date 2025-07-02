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
    SolverControl        solver_control(src.size(), 1e-7 * src.l2_norm());
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

  // -----  block-preconditioner class -----------------
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &System,
                             const PreconditionerTypeaS &ap_S_inv,
                             const PreconditionerTypeM  &ap_M_inv,
                             TimerOutput & computing_timer);

    void
    vmult(TrilinosWrappers::MPI::BlockVector       &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const;

  private:
    const TrilinosWrappers::BlockSparseMatrix &system_matrix;
    const PreconditionerTypeaS                &ap_S_inv;
    const PreconditionerTypeM                 &ap_M_inv;
     TimerOutput & computing_timer;
  };

  // constructor
  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::
    BlockSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &System,
                             const PreconditionerTypeaS &ap_S_inv,
                             const PreconditionerTypeM  &ap_M_inv,
                             TimerOutput & computing_timer)
    : system_matrix(System)
    , ap_S_inv(ap_S_inv)
    , ap_M_inv(ap_M_inv)
    ,computing_timer(computing_timer)
  {}

  template <class PreconditionerTypeaS, class PreconditionerTypeM>
  void
  BlockSchurPreconditioner<PreconditionerTypeaS, PreconditionerTypeM>::vmult(
    TrilinosWrappers::MPI::BlockVector       &dst,
    const TrilinosWrappers::MPI::BlockVector &src) const
  {
    TimerOutput::Scope timer_section(computing_timer, "   Apply preconditioner");
    TrilinosWrappers::MPI::Vector utmp(src.block(1));
    // tmp(complete_index_set(system_matrix.block(1, 1).m())) // there is a
    // problem in parallel
    ap_M_inv.vmult(dst.block(0), src.block(0));
    system_matrix.block(1, 0).residual(utmp, dst.block(0), src.block(1));
    utmp *= -1;
    ap_S_inv.vmult(dst.block(1), utmp);
  }
} // end namespace Preconditioner
#endif
