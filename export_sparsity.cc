// export_sparsity.cc
// MPI-parallel executable to export the lower-triangular sparsity pattern
// of the random field DOF handler as NumPy COO index files.
//
// IMPORTANT: Must be run with the SAME number of MPI ranks as the solver
// (darcy_forward / darcy_adjoint) to ensure identical DOF numbering.
// The DOF ordering depends on the p4est partitioning and Cuthill-McKee
// renumbering, both of which change with the number of MPI ranks.
//
// Usage: mpirun -np <num_procs> ./export_sparsity <parameter_file.json>
//
// Output:
//   rf_sparsity_row_idx.npy  (shape: [nnz, 1], double)
//   rf_sparsity_col_idx.npy  (shape: [nnz, 1], double)
//   rf_A_kappa_values.npy    (shape: [nnz, 1], double)
//
// These files are written to the current working directory.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>
#include <iostream>
#include <string>
#include <vector>

#include "npy.hpp"
#include "parameters.h"

using namespace dealii;

template <int dim>
void
run_export_sparsity(const darcy::Parameters &params)
{
  const unsigned int this_mpi_process =
    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const unsigned int n_mpi_processes =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  ConditionalOStream pcout(std::cout, (this_mpi_process == 0));

  // Build mesh (same geometry as DarcyBase::setup_grid_and_dofs)
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

  // Setup random field DOF handler (same as DarcyBase)
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

  // Build sparsity pattern from FE mesh connectivity.
  // Each rank adds entries from its own cells, then
  // distribute_sparsity_pattern communicates ghost-row entries to the
  // owning rank for completeness.
  AffineConstraints<double> rf_constraints;
  rf_constraints.close();

  DynamicSparsityPattern dsp(n_dofs, n_dofs, locally_relevant);
  DoFTools::make_sparsity_pattern(
    rf_dof_handler, dsp, rf_constraints, false, this_mpi_process);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned,
                                             MPI_COMM_WORLD,
                                             locally_relevant);

  // Assemble A_kappa = G + kappa^2 * M (SPDE precision operator)
  // mirroring DarcyAdjoint::create_rf_laplace_operator()
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

  // Build prior precision operator: Q = G + nugget * M
  const double nugget = params.nugget;
  rf_laplace_matrix.add(nugget, rf_mass_matrix);
  pcout << "Assembled Q = G + " << nugget << " * M" << std::endl;

  // Extract lower-triangular COO entries and A_kappa values for locally
  // owned rows (each DOF is owned by exactly one rank, so no duplicates)
  std::vector<double> local_row_idx;
  std::vector<double> local_col_idx;
  std::vector<double> local_values;

  for (const auto row : locally_owned)
    {
      for (auto it = dsp.begin(row); it != dsp.end(row); ++it)
        {
          if (it->column() <= row)
            {
              local_row_idx.push_back(static_cast<double>(row));
              local_col_idx.push_back(static_cast<double>(it->column()));
              local_values.push_back(rf_laplace_matrix.el(row, it->column()));
            }
        }
    }

  // Gather entry counts from all ranks onto rank 0
  int              local_nnz = static_cast<int>(local_row_idx.size());
  std::vector<int> all_nnz(n_mpi_processes);
  MPI_Gather(
    &local_nnz, 1, MPI_INT, all_nnz.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Compute displacements and total count on rank 0
  std::vector<int> displs(n_mpi_processes, 0);
  int              total_nnz = 0;
  if (this_mpi_process == 0)
    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      {
        displs[i] = total_nnz;
        total_nnz += all_nnz[i];
      }

  // Gather all COO entries and values onto rank 0
  std::vector<double> global_row_idx(this_mpi_process == 0 ? total_nnz : 0);
  std::vector<double> global_col_idx(this_mpi_process == 0 ? total_nnz : 0);
  std::vector<double> global_values(this_mpi_process == 0 ? total_nnz : 0);

  MPI_Gatherv(local_row_idx.data(),
              local_nnz,
              MPI_DOUBLE,
              global_row_idx.data(),
              all_nnz.data(),
              displs.data(),
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

  MPI_Gatherv(local_col_idx.data(),
              local_nnz,
              MPI_DOUBLE,
              global_col_idx.data(),
              all_nnz.data(),
              displs.data(),
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

  MPI_Gatherv(local_values.data(),
              local_nnz,
              MPI_DOUBLE,
              global_values.data(),
              all_nnz.data(),
              displs.data(),
              MPI_DOUBLE,
              0,
              MPI_COMM_WORLD);

  // Write COO indices and A_kappa values as npy files (rank 0 only)
  if (this_mpi_process == 0)
    {
      const auto nnz = static_cast<unsigned long>(total_nnz);
      const std::vector<long unsigned> shape{nnz, 1};
      const bool                       fortran_order = false;

      npy::SaveArrayAsNumpy("rf_sparsity_row_idx.npy",
                            fortran_order,
                            shape.size(),
                            shape.data(),
                            global_row_idx);

      npy::SaveArrayAsNumpy("rf_sparsity_col_idx.npy",
                            fortran_order,
                            shape.size(),
                            shape.data(),
                            global_col_idx);

      npy::SaveArrayAsNumpy("rf_A_kappa_values.npy",
                            fortran_order,
                            shape.size(),
                            shape.data(),
                            global_values);

      pcout << "Number of lower-triangular nonzeros: " << total_nnz
            << std::endl;
      pcout << "Average entries per row: "
            << static_cast<double>(total_nnz) / n_dofs << std::endl;
      pcout << "Wrote rf_sparsity_row_idx.npy, rf_sparsity_col_idx.npy, "
               "and rf_A_kappa_values.npy"
            << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      if (argc != 2)
        {
          std::cerr << "Usage: " << argv[0] << " <parameter_file.json>"
                    << std::endl;
          return 1;
        }

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Parse parameters
      ParameterHandler  prm;
      darcy::Parameters params;
      darcy::Parameters::declare_parameters(prm);
      prm.parse_input(argv[1]);
      params.parse_parameters(prm);

      // Dispatch on spatial dimension
      if (params.spatial_dimension == 2)
        run_export_sparsity<2>(params);
      else
        run_export_sparsity<3>(params);
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
