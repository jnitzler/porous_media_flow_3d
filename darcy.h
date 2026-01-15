// darcy.h

#ifndef DARCY_H
#define DARCY_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// include trilinos headers
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// a timer
#include <deal.II/base/timer.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <string>

// numpy support in c++
#include <memory>

#include "npy.hpp"

using namespace dealii;

// ----- Define the class structure ------------------------------------------
namespace darcy
{

  // ----------- Darcy class ---------------------------
  template <int dim>
  class Darcy
  {
  public:
    explicit Darcy(const unsigned int degree_p);
    void
    run(const std::string &input_path, const std::string &output_path);

  private:
    void
    generate_coordinates();
    void
    run_simulation(const std::string &input_path,
                   const std::string &output_path);
    void
    read_input_npy(const std::string &input_path);
    void
    generate_ref_input();
    void
    read_primary_solution(const std::string &input_path);
    void
    setup_grid_and_dofs();
    void
    setup_grid_and_dofs_adjoint();
    void
    assemble_approx_schur_complement();
    void
    assemble_system();
    void
    solve(const bool solve_adjoint = false);
    void
    output_pvtu(const std::string &output_path) const;
    void
    output_full_velocity_npy(const std::string &output_path);
    void
    output_velocity_at_observation_points_npy(const std::string &output_path);
    void
    output_field_at_observation_points_npy(const std::string &output_path);

    void
    write_data_to_npy(const std::string   &filename,
                      std::vector<double> &data,
                      unsigned int         rows,
                      unsigned int         columns) const;
    void
    read_upstream_gradient_npy(const std::string &input_file_path);
    void
    overwrite_adjoint_rhs();
    void
    final_inner_adjoint_product();
    void
    create_rf_laplace_operator();
    void
    add_prior_gradient_to_adjoint();
    void
    output_gradient_vtu(const std::string &output_path);
    void
    write_adjoint_solution_pvtu(const std::string &output_path);
    void
    write_gradient_to_npy(const std::string &output_path);

    const unsigned int degree_p;
    const unsigned int degree_u;

    parallel::distributed::Triangulation<dim> triangulation; // for problem
    Triangulation<dim>
                    triangulation_obs; // for observation points (serial, small)
    FESystem<dim>   fe;
    DoFHandler<dim> dof_handler;


    // random field setup
    FESystem<dim>   rf_fe_system;
    DoFHandler<dim> rf_dof_handler;

    AffineConstraints<double> constraints;
    AffineConstraints<double> preconditioner_constraints;

    TrilinosWrappers::BlockSparseMatrix system_matrix;
    TrilinosWrappers::BlockSparseMatrix precondition_matrix;

    TrilinosWrappers::MPI::BlockVector solution;
    TrilinosWrappers::MPI::BlockVector system_rhs;
    TrilinosWrappers::MPI::BlockVector temp_vec;
    TrilinosWrappers::MPI::BlockVector solution_distributed;

    TrilinosWrappers::BlockSparseMatrix block_mass_matrix;
    std::vector<double>                 grad_log_lik_x_distributed;
    std::vector<double>                 grad_log_lik_x;
    FullMatrix<double>
      grad_pde_x; // New PDE gradient that needs to be implemented

    // random field stuff (distributed with ghost values for parallel access)
    TrilinosWrappers::MPI::Vector x_vec; // the input vector for the simulation
    TrilinosWrappers::MPI::Vector
      x_vec_distributed; // the input vector for the simulation
    TrilinosWrappers::MPI::Vector mean_rf; // mean of the random field

    // Index sets for the random field DOFs
    IndexSet rf_locally_owned;
    IndexSet rf_locally_relevant;

    // Vector<double> random_field_vec;
    std::vector<Point<dim>> spatial_coordinates;

    ConditionalOStream  pcout;
    TimerOutput         computing_timer;  // timer for sub programs
    std::vector<double> adjoint_data_vec; // adjoint data vector with partial
                                          // derivative of likelihood
    std::vector<double> grad_log_lik_x_partial_distributed;

    std::vector<std::vector<double>>
      data_vec; // general input data which will be split to actual data
    TrilinosWrappers::MPI::BlockVector
      solution_primary_problem; // distributed solution vector of the primary
                                // problem
    TrilinosWrappers::MPI::BlockVector solution_primary_distributed;

    TrilinosWrappers::SparseMatrix rf_laplace_matrix; // laplace matrix for the
                                                      // random field

    void
    setup_system_matrix(
      const std::vector<IndexSet> &system_partitioning,
      const std::vector<IndexSet> &system_relevant_partitioning);

    void
    setup_approx_schur_complement(
      const std::vector<IndexSet> &system_partitioning,
      const std::vector<IndexSet> &system_relevant_partitioning);
  };
} // namespace darcy
#endif