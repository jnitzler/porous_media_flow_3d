// darcy.h
// Main header for the parallel Darcy flow solver with adjoint capability.
// Solves mixed formulation: K^{-1} u + grad(p) = 0, div(u) = f
// with random permeability field K = exp(x) for uncertainty quantification.

#ifndef DARCY_H
#define DARCY_H

// === deal.II base includes ===
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

// === Distributed triangulation and grid ===
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

// === Finite elements and DOF handling ===
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

// === Linear algebra (serial) ===
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

// === Trilinos parallel linear algebra ===
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// === Numerics (output, assembly helpers) ===
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// === Standard library ===
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <string>

// === External: NumPy file I/O ===
#include "npy.hpp"

using namespace dealii;

namespace darcy
{
  // ===========================================================================
  // Darcy class: MPI-parallel mixed finite element solver for Darcy flow.
  //
  // Features:
  //   - Mixed FE: Raviart-Thomas-like velocity (Q_{k+1}) + pressure (Q_k)
  //   - Random permeability field K = exp(x) on separate scalar FE space
  //   - Adjoint solver for gradient-based optimization / inverse problems
  //   - Weak boundary conditions via Nitsche's method
  //   - Block Schur complement preconditioner for saddle-point system
  // ===========================================================================
  template <int dim>
  class Darcy
  {
  public:
    // Constructor: degree_p is the polynomial degree for pressure (velocity =
    // degree_p + 1)
    explicit Darcy(const unsigned int degree_p);

    // Main entry point for forward/adjoint simulation
    void
    run(const std::string &input_path, const std::string &output_path);

  private:
    // -------------------------------------------------------------------------
    // Setup methods
    // -------------------------------------------------------------------------
    void
    setup_grid_and_dofs(); // Create mesh, distribute DOFs, init matrices
    void
    setup_grid_and_dofs_adjoint(); // Adjoint-specific setup (if different)
    void
    setup_system_matrix(const std::vector<IndexSet> &partitioning,
                        const std::vector<IndexSet> &relevant_partitioning);
    void
    setup_approx_schur_complement(
      const std::vector<IndexSet> &partitioning,
      const std::vector<IndexSet> &relevant_partitioning);

    // -------------------------------------------------------------------------
    // Assembly methods
    // -------------------------------------------------------------------------
    void
    assemble_system(); // Assemble Darcy saddle-point system
    void
    assemble_approx_schur_complement(); // Assemble preconditioner matrix

    // -------------------------------------------------------------------------
    // Solver
    // -------------------------------------------------------------------------
    void
    solve(const bool solve_adjoint = false); // GMRES with block preconditioner

    // -------------------------------------------------------------------------
    // Input methods
    // -------------------------------------------------------------------------
    void
    read_input_npy(
      const std::string &input_path); // Read random field from .npy
    void
    generate_ref_input(); // Generate test random field
    void
    generate_coordinates(); // Setup observation points
    void
    read_primary_solution(
      const std::string &input_path); // Load forward solution (adjoint)
    void
    read_upstream_gradient_npy(
      const std::string &input_file_path); // Load dL/dy (adjoint)

    // -------------------------------------------------------------------------
    // Output methods
    // -------------------------------------------------------------------------
    void
    output_pvtu(const std::string &output_path) const; // VTU for ParaView
    void
    output_full_velocity_npy(
      const std::string &output_path); // Full solution to .npy
    void
    output_velocity_at_observation_points_npy(const std::string &output_path);
    void
    output_field_at_observation_points_npy(const std::string &output_path);
    void
    write_data_to_npy(const std::string   &filename,
                      std::vector<double> &data,
                      unsigned int         rows,
                      unsigned int         columns) const;

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
    // Member variables
    // =========================================================================

    // --- Polynomial degrees ---
    const unsigned int degree_p; // Pressure degree
    const unsigned int degree_u; // Velocity degree (= degree_p + 1)

    // --- Mesh ---
    parallel::distributed::Triangulation<dim>
      triangulation; // Main mesh (MPI-parallel)
    Triangulation<dim>
      triangulation_obs; // Observation mesh (serial, for point evaluation)

    // --- Finite elements and DOF handlers ---
    FESystem<dim>   fe;          // Mixed FE: [Q_{degree_u}]^dim x Q_{degree_p}
    DoFHandler<dim> dof_handler; // DOF handler for velocity-pressure

    FESystem<dim>   rf_fe_system;   // Scalar FE for random field (Q1)
    DoFHandler<dim> rf_dof_handler; // DOF handler for random field

    // --- Constraints ---
    AffineConstraints<double> constraints;                // For system matrix
    AffineConstraints<double> preconditioner_constraints; // For preconditioner

    // --- System matrices ---
    TrilinosWrappers::BlockSparseMatrix system_matrix; // Saddle-point matrix
    TrilinosWrappers::BlockSparseMatrix
      precondition_matrix; // Schur preconditioner
    TrilinosWrappers::BlockSparseMatrix
      block_mass_matrix; // For adjoint (if needed)

    // --- Solution vectors ---
    TrilinosWrappers::MPI::BlockVector solution; // Current solution [u, p]
    TrilinosWrappers::MPI::BlockVector
      solution_distributed;                        // With ghost values
    TrilinosWrappers::MPI::BlockVector system_rhs; // Right-hand side
    TrilinosWrappers::MPI::BlockVector temp_vec;   // Temporary storage

    // --- Adjoint-specific vectors ---
    TrilinosWrappers::MPI::BlockVector
      solution_primary_problem; // Forward solution
    TrilinosWrappers::MPI::BlockVector
      solution_primary_distributed; // With ghosts
    std::vector<double>
      grad_log_lik_x_distributed;       // Gradient (local contributions)
    std::vector<double> grad_log_lik_x; // Gradient (gathered on all ranks)
    FullMatrix<double>  grad_pde_x;     // Reserved for future use

    // --- Random field vectors ---
    TrilinosWrappers::MPI::Vector x_vec; // Log-permeability (owned DOFs)
    TrilinosWrappers::MPI::Vector
      x_vec_distributed;                    // Log-permeability (with ghosts)
    TrilinosWrappers::MPI::Vector  mean_rf; // Prior mean (typically zero)
    TrilinosWrappers::SparseMatrix rf_laplace_matrix; // GMRF precision matrix L

    // --- Random field index sets ---
    IndexSet rf_locally_owned;    // DOFs owned by this MPI rank
    IndexSet rf_locally_relevant; // DOFs needed (owned + ghost)

    // --- Observation data ---
    std::vector<Point<dim>> spatial_coordinates; // Observation locations
    std::vector<double>     adjoint_data_vec; // Upstream gradient dL/dy (flat)
    std::vector<std::vector<double>>
                        data_vec; // Upstream gradient [point][component]
    std::vector<double> grad_log_lik_x_partial_distributed; // Partial gradient

    // --- I/O and timing ---
    ConditionalOStream pcout;           // Only rank 0 prints
    TimerOutput        computing_timer; // Performance profiling
  };

} // namespace darcy

#endif // DARCY_H