// darcy_adjoint.h
// Adjoint Darcy solver class for gradient computation.
// Implements the adjoint simulation for gradient-based optimization.

#ifndef DARCY_ADJOINT_H
#define DARCY_ADJOINT_H

#include "darcy_base.h"

namespace darcy
{
  // ===========================================================================
  // DarcyAdjoint class: Adjoint Darcy solver for gradient computation
  // ===========================================================================
  template <int dim>
  class DarcyAdjoint : public DarcyBase<dim>
  {
  public:
    // Constructor
    explicit DarcyAdjoint(const unsigned int degree_p);

    // Main entry point for adjoint simulation
    void
    run(const std::string &input_path, const std::string &output_path) override;

  private:
    // -------------------------------------------------------------------------
    // Adjoint-specific input methods
    // -------------------------------------------------------------------------
    void
    read_primary_solution(
      const std::string &input_path); // Load forward solution
    void
    read_upstream_gradient_npy(
      const std::string &input_file_path); // Load dL/dy

    // -------------------------------------------------------------------------
    // Adjoint-specific methods
    // -------------------------------------------------------------------------
    void
    overwrite_adjoint_rhs(); // Build RHS from upstream gradient
    void
    final_inner_adjoint_product(); // Compute dL/dx = -lambda^T (dA/dx) u
    void
    create_rf_laplace_operator(); // GMRF prior precision matrix
    void
    add_prior_gradient_to_adjoint(); // Add prior regularization to gradient

    // -------------------------------------------------------------------------
    // Adjoint-specific output methods
    // -------------------------------------------------------------------------
    void
    write_adjoint_solution_pvtu(const std::string &output_path);
    void
    output_gradient_pvtu(const std::string &output_path);
    void
    write_gradient_to_npy(const std::string &output_path);

    // -------------------------------------------------------------------------
    // Simulation driver
    // -------------------------------------------------------------------------
    void
    run_simulation(const std::string &input_path,
                   const std::string &output_path);

    // =========================================================================
    // Adjoint-specific member variables
    // =========================================================================

    // --- Adjoint-specific vectors ---
    TrilinosWrappers::MPI::BlockVector
      solution_primary_problem; // Forward solution
    TrilinosWrappers::MPI::BlockVector
      solution_primary_distributed; // With ghosts
    std::vector<double>
      grad_log_lik_x_distributed;       // Gradient (local contributions)
    std::vector<double> grad_log_lik_x; // Gradient (gathered on all ranks)
    FullMatrix<double>  grad_pde_x;     // Reserved for future use

    // --- Adjoint observation data ---
    std::vector<double> adjoint_data_vec; // Upstream gradient dL/dy (flat)
    std::vector<std::vector<double>>
                        data_vec; // Upstream gradient [point][component]
    std::vector<double> grad_log_lik_x_partial_distributed; // Partial gradient
  };

  // Constructor implementation
  template <int dim>
  DarcyAdjoint<dim>::DarcyAdjoint(const unsigned int degree_p)
    : DarcyBase<dim>(degree_p)
  {}

} // namespace darcy

#endif // DARCY_ADJOINT_H
