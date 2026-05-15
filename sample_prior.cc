// sample_prior.cc
// Standalone, one-shot diagnostic executable used to visualize the SPDE prior
// field. Not part of the inversion pipeline: it shares the parameter struct
// only for mesh/FE settings and exposes its own --nugget and --tau CLI flags
// (the JSON "nugget" entry is intentionally not consulted here).
//
// Draws samples from the SPDE prior with covariance Q^{-1} M Q^{-1}, where
// Q = G + nugget*M on the same eccentric hyper-shell mesh used by the
// forward/adjoint solvers.
//
// Sampling method (element-wise mass Cholesky + CG solve):
//   1. Generate w ~ N(0, M) by assembling element-wise: for each cell compute
//      M_e, dense Cholesky M_e = L_e L_e^T, then w_e = L_e v_e with
//      v_e ~ N(0, I). Assembly gives w ~ N(0, sum M_e) = N(0, M).
//   2. Solve Q z = w using CG. Then z ~ N(0, Q^{-1} M Q^{-1}) (SPDE prior).
//
// Usage:
//   mpirun -np <num_procs> ./sample_prior <parameter_file.json>
//       --num-samples <N> --output-dir <dir> [--seed <S>]
//
// Output:
//   {output_dir}/{prefix}sample_0000.npy   (shape: [n_dofs, 1], double)
//   {output_dir}/{prefix}sample_0000.pvtu  (ParaView visualization)

#include <cmath>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "npy.hpp"
#include "parameters.h"

using namespace dealii;

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct SamplePriorArgs
{
  std::string  param_file;
  unsigned int num_samples = 1;
  std::string  output_dir  = "output/samples";
  unsigned int seed        = 42;
  double       tau         = 1.0;
  double       nugget      = 1e-3;
};

SamplePriorArgs
parse_args(int argc, char *argv[])
{
  SamplePriorArgs args;

  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i)
    {
      std::string arg(argv[i]);
      if (arg == "--num-samples" && i + 1 < argc)
        {
          args.num_samples = std::stoul(argv[++i]);
        }
      else if (arg == "--output-dir" && i + 1 < argc)
        {
          args.output_dir = argv[++i];
        }
      else if (arg == "--seed" && i + 1 < argc)
        {
          args.seed = std::stoul(argv[++i]);
        }
      else if (arg == "--tau" && i + 1 < argc)
        {
          args.tau = std::stod(argv[++i]);
        }
      else if (arg == "--nugget" && i + 1 < argc)
        {
          args.nugget = std::stod(argv[++i]);
        }
      else if (arg[0] != '-')
        {
          positional.push_back(arg);
        }
      else
        {
          throw std::runtime_error("Unknown argument: " + arg);
        }
    }

  if (positional.empty())
    throw std::runtime_error(
      "Usage: sample_prior <parameter_file.json> "
      "--num-samples <N> --output-dir <dir> [--seed <S>] "
      "[--tau <T>] [--nugget <N>]");

  args.param_file = positional[0];
  return args;
}

// ---------------------------------------------------------------------------
// Main sampling routine
// ---------------------------------------------------------------------------
template <int dim>
void
run_sample_prior(const darcy::Parameters &params, const SamplePriorArgs &args)
{
  const unsigned int this_mpi_process =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int n_mpi_processes =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  ConditionalOStream pcout(std::cout, (this_mpi_process == 0));
  TimerOutput timer(pcout, TimerOutput::summary, TimerOutput::wall_times);

  // ---- Mesh setup (same geometry as DarcyBase) ----
  timer.enter_subsection("Setup mesh");
  parallel::distributed::Triangulation<dim> triangulation(
    MPI_COMM_WORLD,
    typename Triangulation<dim>::MeshSmoothing(
      Triangulation<dim>::smoothing_on_refinement |
      Triangulation<dim>::smoothing_on_coarsening));

  Point<dim> inner_center;
  Point<dim> outer_center;
  if constexpr (dim == 2)
    {
      inner_center = Point<dim>(0.0, 0.1);
      outer_center = Point<dim>(0.0, 0.0);
    }
  else if constexpr (dim == 3)
    {
      inner_center = Point<dim>(0.0, 0.1, 0.25);
      outer_center = Point<dim>(0.0, 0.0, 0.0);
    }

  double       inner_radius = 0.3;
  double       outer_radius = 1.0;
  unsigned int n_cells      = 12;

  GridGenerator::eccentric_hyper_shell(triangulation,
                                       inner_center,
                                       outer_center,
                                       inner_radius,
                                       outer_radius,
                                       n_cells);
  triangulation.refine_global(params.refinement_level);
  timer.leave_subsection();

  // ---- RF DOF handler ----
  timer.enter_subsection("Setup DOFs");
  FESystem<dim>   rf_fe_system(FE_Q<dim>(params.degree_rf), 1);
  DoFHandler<dim> rf_dof_handler(triangulation);
  rf_dof_handler.distribute_dofs(rf_fe_system);
  DoFRenumbering::Cuthill_McKee(rf_dof_handler);

  const types::global_dof_index n_dofs = rf_dof_handler.n_dofs();
  const IndexSet locally_owned         = rf_dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant =
    DoFTools::extract_locally_relevant_dofs(rf_dof_handler);

  pcout << "Number of random field DOFs: " << n_dofs << std::endl;
  pcout << "Number of MPI processes: " << n_mpi_processes << std::endl;

  AffineConstraints<double> rf_constraints;
  rf_constraints.close();
  timer.leave_subsection();

  // ---- Assemble Q = G + nugget * M ----
  timer.enter_subsection("Assemble Q");
  TrilinosWrappers::SparsityPattern sp_rf(locally_owned,
                                          locally_owned,
                                          locally_relevant,
                                          MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(
    rf_dof_handler, sp_rf, rf_constraints, false, this_mpi_process);
  sp_rf.compress();

  TrilinosWrappers::SparseMatrix rf_laplace_matrix;
  rf_laplace_matrix.reinit(sp_rf);

  TrilinosWrappers::SparseMatrix rf_mass_matrix;
  rf_mass_matrix.reinit(sp_rf);

  const QGauss<dim>            quadrature(params.fe_degree + 2);
  const MappingQ<dim>          mapping(2);
  const Function<dim, double> *coefficient = nullptr;

  MatrixCreator::create_laplace_matrix(mapping,
                                       rf_dof_handler,
                                       quadrature,
                                       rf_laplace_matrix,
                                       coefficient,
                                       rf_constraints);

  MatrixCreator::create_mass_matrix(mapping,
                                    rf_dof_handler,
                                    quadrature,
                                    rf_mass_matrix,
                                    coefficient,
                                    rf_constraints);

  // Q = G + nugget * M
  const double nugget = args.nugget;
  rf_laplace_matrix.add(nugget, rf_mass_matrix);
  pcout << "Assembled Q = G + " << nugget << " * M  (tau = " << args.tau << ")"
        << std::endl;
  timer.leave_subsection();

  // ---- Setup CG preconditioner ----
  timer.enter_subsection("Setup preconditioner");
  TrilinosWrappers::PreconditionAMG                 preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.elliptic              = true;
  amg_data.higher_order_elements = (params.degree_rf > 1);
  preconditioner.initialize(rf_laplace_matrix, amg_data);
  timer.leave_subsection();

  // ---- Create output directory ----
  if (this_mpi_process == 0)
    std::filesystem::create_directories(args.output_dir);
  MPI_Barrier(MPI_COMM_WORLD);

  // ---- FEValues for element-wise random RHS generation ----
  const unsigned int dofs_per_cell = rf_fe_system.n_dofs_per_cell();

  FEValues<dim> fe_values(mapping,
                          rf_fe_system,
                          quadrature,
                          update_values | update_gradients | update_JxW_values);

  // ---- Gather infrastructure for npy output ----
  IndexSet all_on_rank0(n_dofs);
  if (this_mpi_process == 0)
    all_on_rank0.add_range(0, n_dofs);

  // ---- Sample loop ----
  const std::string prefix = params.output_prefix;

  pcout << "Drawing " << args.num_samples
        << " prior samples (seed=" << args.seed << ")..." << std::endl;

  timer.enter_subsection("Sampling");
  for (unsigned int s = 0; s < args.num_samples; ++s)
    {
      // Step 1: Generate w ~ N(0, M) via element-wise Cholesky of the mass
      // matrix. SPDE: (kappa^2 - Delta)x = W, FEM weak form gives
      // Q x = b where b ~ N(0, M), so Cov(x) = Q^{-1} M Q^{-1}.
      TrilinosWrappers::MPI::Vector w(locally_owned, MPI_COMM_WORLD);

      std::mt19937_64 gen(args.seed + s * n_mpi_processes + this_mpi_process);
      std::normal_distribution<double> dist(0.0, 1.0);

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);
      Vector<double>     v(dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : rf_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            cell_matrix = 0;

            // Local mass matrix M_e (symmetric, fill full matrix)
            for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                for (unsigned int j = 0; j <= i; ++j)
                  {
                    const double val = fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) *
                                       fe_values.JxW(q);
                    cell_matrix(i, j) += val;
                    if (i != j)
                      cell_matrix(j, i) += val;
                  }

            // Dense Cholesky in-place: cell_matrix -> L_M (lower triangle)
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                double sum = cell_matrix(j, j);
                for (unsigned int k = 0; k < j; ++k)
                  sum -= cell_matrix(j, k) * cell_matrix(j, k);
                cell_matrix(j, j) = std::sqrt(sum);
                for (unsigned int i = j + 1; i < dofs_per_cell; ++i)
                  {
                    double s = cell_matrix(i, j);
                    for (unsigned int k = 0; k < j; ++k)
                      s -= cell_matrix(i, k) * cell_matrix(j, k);
                    cell_matrix(i, j) = s / cell_matrix(j, j);
                  }
              }

            // v ~ N(0, I)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              v(i) = dist(gen);

            // cell_rhs = L_M * v (lower triangular multiply)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                double s = 0;
                for (unsigned int j = 0; j <= i; ++j)
                  s += cell_matrix(i, j) * v(j);
                cell_rhs(i) = s;
              }

            // Assemble into global vector
            cell->get_dof_indices(local_dof_indices);
            rf_constraints.distribute_local_to_global(cell_rhs,
                                                      local_dof_indices,
                                                      w);
          }
      w.compress(VectorOperation::add);

      // Step 2: Solve Q z = w. Cov(z) = Q^{-1} M Q^{-1} (SPDE covariance).
      TrilinosWrappers::MPI::Vector solution(locally_owned, MPI_COMM_WORLD);
      SolverControl                 solver_control(5000, 1e-10 * w.l2_norm());
      SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
      solver.solve(rf_laplace_matrix, solution, w, preconditioner);

      // Subtract mean to center samples around zero
      const double global_mean = solution.mean_value();
      solution.add(-global_mean);

      // Scale by 1/tau: x ~ N(0, (tau^2 Q)^{-1}) = (1/tau) * N(0, Q^{-1})
      if (args.tau != 1.0)
        solution /= args.tau;

      pcout << "  Sample " << s << ": CG converged in "
            << solver_control.last_step() << " iterations" << std::endl;

      // Write VTU output for ParaView
      TrilinosWrappers::MPI::Vector solution_ghosted(locally_owned,
                                                     locally_relevant,
                                                     MPI_COMM_WORLD);
      solution_ghosted = solution;

      DataOut<dim>          data_out;
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out.set_flags(flags);
      data_out.add_data_vector(rf_dof_handler, solution_ghosted, "sample");
      data_out.build_patches(mapping, 2, DataOut<dim>::curved_boundary);

      std::ostringstream vtu_name;
      vtu_name << prefix << "sample_" << std::setfill('0') << std::setw(4) << s;

      const std::string vtu_path = args.output_dir + "/";
      data_out.write_vtu_with_pvtu_record(
        vtu_path, vtu_name.str(), 0, MPI_COMM_WORLD, 1);

      // Write .npy on rank 0
      TrilinosWrappers::MPI::Vector gathered(all_on_rank0, MPI_COMM_WORLD);
      gathered = solution;

      if (this_mpi_process == 0)
        {
          std::vector<double> data(n_dofs);
          for (unsigned int i = 0; i < n_dofs; ++i)
            data[i] = gathered[i];

          std::ostringstream filename;
          filename << args.output_dir << "/" << prefix << "sample_"
                   << std::setfill('0') << std::setw(4) << s << ".npy";

          const std::vector<long unsigned> shape{
            static_cast<unsigned long>(n_dofs), 1};
          const bool fortran_order = false;
          npy::SaveArrayAsNumpy(
            filename.str(), fortran_order, shape.size(), shape.data(), data);
        }
    }
  timer.leave_subsection();

  pcout << "Wrote " << args.num_samples << " samples to " << args.output_dir
        << "/" << std::endl;
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      SamplePriorArgs args = parse_args(argc, argv);

      // Parse parameters
      ParameterHandler  prm;
      darcy::Parameters params;
      darcy::Parameters::declare_parameters(prm);
      prm.parse_input(args.param_file);
      params.parse_parameters(prm);

      // Dispatch on spatial dimension
      if (params.spatial_dimension == 2)
        run_sample_prior<2>(params, args);
      else
        run_sample_prior<3>(params, args);
    }
  catch (std::exception &exc)
    {
      std::cerr << "Exception on processor "
                << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << ": "
                << exc.what() << std::endl;
      return 1;
    }

  return 0;
}
