#include "darcy.h"

#include <cstddef> // std::size_t

#include "darcy_general.h"

namespace darcy // same namespace and in header file
{
  // --- output velocity at observation points to numpy array ---------
  template <int dim>
  void
  Darcy<dim>::output_velocity_at_observation_points_npy(
    const std::string &output_path)
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Remote point evaluation");

    MappingQ<dim> dummy_mapping(1);

    solution_distributed = solution;

    // Use the old API that works correctly
    Utilities::MPI::RemotePointEvaluation<dim> evaluation_cache(1.0e-9);
    const auto data_array = VectorTools::point_values<dim>(dummy_mapping,
                                                           dof_handler,
                                                           solution_distributed,
                                                           spatial_coordinates,
                                                           evaluation_cache);

    // Only rank 0 writes the output file
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const unsigned int  n_points = data_array.size();
        std::vector<double> output_data(dim * n_points);

        // Reorder: [all u_1, all u_2, all u_3]
        for (unsigned int d = 0; d < dim; ++d)
          {
            for (unsigned int i = 0; i < n_points; ++i)
              {
                output_data[d * n_points + i] = data_array[i][d];
              }
          }
        const std::string filename        = output_path + "_sol.npy";
        const std::string filename_coords = output_path + "_coords.npy";
        write_data_to_npy(filename, output_data, output_data.size(), 1);
      }
  }


  // ----------------- output_npy -----------------
  template <int dim>
  void
  Darcy<dim>::output_full_velocity_npy(const std::string &output_path)
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Output full solution npy");

    solution_distributed = solution;

    const unsigned int n_dofs = dof_handler.n_dofs();
    const unsigned int n_u    = solution.block(0).size();

    // Create local buffer - store locally owned values, zeros elsewhere
    std::vector<double> local_solution(n_dofs, 0.0);

    // Fill velocity block (global indices 0 to n_u-1)
    for (const auto idx : solution.block(0).locally_owned_elements())
      local_solution[idx] = solution.block(0)[idx];

    // Fill pressure block (global indices n_u to n_dofs-1)
    for (const auto idx : solution.block(1).locally_owned_elements())
      local_solution[n_u + idx] = solution.block(1)[idx];

    // Single MPI_Reduce to gather everything to rank 0
    std::vector<double> full_solution;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      full_solution.resize(n_dofs, 0.0);
    else
      full_solution.resize(1); // Non-root ranks need valid pointer

    MPI_Reduce(local_solution.data(),
               full_solution.data(),
               n_dofs,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    // Write the gathered solution on rank 0
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string file_path = output_path + "_solution_full.npy";
        pcout << "Writing full solution to file: " << file_path << std::endl;
        pcout << "Size of full solution: " << full_solution.size() << std::endl;
        pcout << "Number of dofs: " << n_dofs << std::endl;
        write_data_to_npy(file_path, full_solution, full_solution.size(), 1);
      }
  }

  // ----------------- output pvtu -----------------
  template <int dim>
  void
  Darcy<dim>::output_pvtu(const std::string &output_path) const
  {
    const std::size_t found    = output_path.find_last_of("/\\");
    const std::string filename = output_path.substr(found + 1) + "_solution";
    const std::string stripped_path = output_path.substr(0, found + 1);

    // define the solution vector
    std::vector<std::string> solution_names(dim, "u");
    solution_names.emplace_back("p");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.emplace_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true; // Use standard VTK cells
    data_out.set_flags(flags);
    data_out.add_data_vector(dof_handler,
                             solution_distributed,
                             solution_names,
                             interpretation);

    // define the subdomains
    Vector<float> subdomain(triangulation.n_active_cells());
    unsigned int  cell_index = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
      {
        subdomain(cell_index) = cell->subdomain_id();
        ++cell_index;
      }
    data_out.add_data_vector(subdomain, "subdomain");

    // build the patches
    MappingQ<dim> mapping(1); // nonlinear mapping
    data_out.build_patches(mapping, degree_u, DataOut<dim>::curved_inner_cells);

    constexpr unsigned int n_digits_counter = 2;
    constexpr unsigned int cycle = 0; // a counter for iterative output
    pcout << "stripped_path for solution: " << stripped_path << std::endl;
    data_out.write_vtu_with_pvtu_record(
      stripped_path, filename, cycle, MPI_COMM_WORLD, n_digits_counter);

    pcout << "Written pvtu for the solution!" << std::endl;

    // define the random field
    const std::string filename_rf =
      output_path.substr(found + 1) + "_random_field";
    const std::string stripped_path_rf = output_path.substr(0, found + 1);

    DataOut<dim> data_out_rf;
    data_out_rf.set_flags(flags);

    // Output the raw log-field for debugging
    std::string random_field_names_log = "log_k";
    data_out_rf.add_data_vector(rf_dof_handler,
                                x_vec_distributed,
                                random_field_names_log);

    // Also output the exponentiated field k = exp(x_vec)
    // Create an owned-only vector, fill it, then assign to ghosted vector
    TrilinosWrappers::MPI::Vector rf_owned(rf_locally_owned, MPI_COMM_WORLD);
    for (const auto i : rf_locally_owned)
      rf_owned[i] = std::exp(x_vec_distributed[i]);
    rf_owned.compress(VectorOperation::insert);

    TrilinosWrappers::MPI::Vector rf(rf_locally_owned,
                                     rf_locally_relevant,
                                     MPI_COMM_WORLD);
    rf = rf_owned; // Assignment to ghosted vector updates ghost values


    std::string random_field_names = "k";
    data_out_rf.add_data_vector(rf_dof_handler, rf, random_field_names);

    // Use curved cells with subdivision matching the FE degree for nice output.
    // NOTE: In ParaView, keep "Nonlinear Subdivision Level" at 0 to avoid
    // artifacts - ParaView's subdivision algorithm doesn't match FEM
    // interpolation.
    data_out_rf.build_patches(mapping, 1, DataOut<dim>::curved_inner_cells);
    data_out_rf.write_vtu_with_pvtu_record(
      stripped_path_rf, filename_rf, cycle, MPI_COMM_WORLD, n_digits_counter);
  }

  // ----------------- run methods -----------------
  template <int dim>
  void
  Darcy<dim>::run(const std::string &input_path, const std::string &output_path)
  {
    run_simulation(input_path, output_path);

    computing_timer.print_summary();
    computing_timer.reset();
    pcout << std::endl;
  }

  template <int dim>
  void
  Darcy<dim>::run_simulation(const std::string &input_path,
                             const std::string &output_path)
  {
    setup_grid_and_dofs();
    read_input_npy(input_path);
    // generate_ref_input(); // TODO: should be removed in production
    generate_coordinates();
    assemble_approx_schur_complement();
    assemble_system();
    solve();

    // Copy solution to ghosted vector for output
    solution_distributed = solution;
    x_vec_distributed    = x_vec;

    // output the results
    output_pvtu(output_path);
    output_full_velocity_npy(output_path);
    output_velocity_at_observation_points_npy(output_path);

  } // end run simulation

} // namespace darcy

// ----------------- main -----------------
int
main(int argc, char *argv[])
{
  std::string input_file_path  = argv[1];
  std::string output_file_path = argv[2];

  try
    {
      using namespace darcy;
      // Third argument = 1 disables TBB multi-threading for reproducibility.
      // Even with 1 MPI rank, TBB can use multiple threads for assembly/solver
      // causing non-deterministic floating-point summation order.
      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, numbers::invalid_unsigned_int);
      const unsigned int fe_degree = 1;
      Darcy<2>           mixed_laplace_problem(fe_degree);
      mixed_laplace_problem.run(input_file_path, output_file_path);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
