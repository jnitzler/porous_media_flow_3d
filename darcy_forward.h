// darcy_forward.h
// Forward Darcy flow solver class.
// Implements the forward simulation for the Darcy flow problem.

#ifndef DARCY_FORWARD_H
#define DARCY_FORWARD_H

#include <deal.II/base/mpi_remote_point_evaluation.h>

#include "darcy_base.h"

namespace darcy
{
  // ===========================================================================
  // DarcyForward class: Forward Darcy flow solver
  // ===========================================================================
  template <int dim>
  class DarcyForward : public DarcyBase<dim>
  {
  public:
    // Constructor
    explicit DarcyForward(const unsigned int degree_p);

    // Main entry point for forward simulation
    void
    run(const Parameters &params) override;

  private:
    // -------------------------------------------------------------------------
    // Forward-specific output methods
    // -------------------------------------------------------------------------
    void
    output_pvtu() const; // VTU for ParaView
    void
    output_full_velocity_npy(); // Full solution to .npy
    void
    output_velocity_at_observation_points_npy();

    // -------------------------------------------------------------------------
    // Simulation driver
    // -------------------------------------------------------------------------
    void
    run_simulation();
  };

  // Constructor implementation
  template <int dim>
  DarcyForward<dim>::DarcyForward(const unsigned int degree_p)
    : DarcyBase<dim>(degree_p)
  {}

  // ===========================================================================
  // Template implementations
  // ===========================================================================

  // Evaluate velocity at observation points and write to "_sol.npy".
  // Output format: [all u_1, all u_2, ...] - velocity components stacked.
  // Only rank 0 writes the file.
  template <int dim>
  void
  DarcyForward<dim>::output_velocity_at_observation_points_npy()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Remote point evaluation");

    MappingQ<dim> dummy_mapping(2);

    this->solution_distributed = this->solution;

    // Use the old API that works correctly
    Utilities::MPI::RemotePointEvaluation<dim> evaluation_cache(1.0e-9);
    const auto data_array = VectorTools::point_values<dim>(dummy_mapping,
                                                           this->dof_handler,
                                                           this->solution_distributed,
                                                           this->spatial_coordinates,
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
        const std::string filename = 
          this->params.output_directory + "/" + this->params.output_prefix + "sol.npy";
        const std::string filename_coords = 
          this->params.output_directory + "/" + this->params.output_prefix + "coords.npy";
        this->write_data_to_npy(filename, output_data, output_data.size(), 1);
      }
  }


  // Write full solution (velocity + pressure) to "_solution_full.npy".
  // Gathers distributed solution to rank 0 using single MPI_Reduce.
  // Only rank 0 writes the file.
  template <int dim>
  void
  DarcyForward<dim>::output_full_velocity_npy()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Output full solution npy");

    this->solution_distributed = this->solution;

    const unsigned int n_dofs = this->dof_handler.n_dofs();
    const unsigned int n_u    = this->solution.block(0).size();

    // Create local buffer - store locally owned values, zeros elsewhere
    std::vector<double> local_solution(n_dofs, 0.0);

    // Fill velocity block (global indices 0 to n_u-1)
    for (const auto idx : this->solution.block(0).locally_owned_elements())
      local_solution[idx] = this->solution.block(0)[idx];

    // Fill pressure block (global indices n_u to n_dofs-1)
    for (const auto idx : this->solution.block(1).locally_owned_elements())
      local_solution[n_u + idx] = this->solution.block(1)[idx];

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
        const std::string file_path = 
          this->params.output_directory + "/" + this->params.output_prefix + "solution_full.npy";
        this->pcout << "Writing full solution to file: " << file_path << std::endl;
        this->pcout << "Size of full solution: " << full_solution.size() << std::endl;
        this->pcout << "Number of dofs: " << n_dofs << std::endl;
        this->write_data_to_npy(file_path, full_solution, full_solution.size(), 1);
      }
  }

  // Write solution and random field to PVTU files for ParaView visualization.
  // Outputs: "_solution.pvtu" with velocity/pressure, "_random_field.pvtu" with
  // log_k and k.
  template <int dim>
  void
  DarcyForward<dim>::output_pvtu() const
  {
    const std::string filename = this->params.output_prefix + "solution";
    const std::string stripped_path = this->params.output_directory + "/";

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
    data_out.add_data_vector(this->dof_handler,
                             this->solution_distributed,
                             solution_names,
                             interpretation);

    // define the subdomains
    Vector<float> subdomain(this->triangulation.n_active_cells());
    unsigned int  cell_index = 0;
    for (const auto &cell : this->triangulation.active_cell_iterators())
      {
        subdomain(cell_index) = cell->subdomain_id();
        ++cell_index;
      }
    data_out.add_data_vector(subdomain, "subdomain");

    // build the patches
    MappingQ<dim> mapping(2); // nonlinear mapping
    data_out.build_patches(mapping, this->degree_u, DataOut<dim>::curved_inner_cells);

    constexpr unsigned int n_digits_counter = 2;
    constexpr unsigned int cycle = 0; // a counter for iterative output
    this->pcout << "stripped_path for solution: " << stripped_path << std::endl;
    data_out.write_vtu_with_pvtu_record(
      stripped_path, filename, cycle, MPI_COMM_WORLD, n_digits_counter);

    this->pcout << "Written pvtu for the solution!" << std::endl;

    // define the random field
    const std::string filename_rf = this->params.output_prefix + "random_field";
    const std::string stripped_path_rf = this->params.output_directory + "/";

    DataOut<dim> data_out_rf;
    data_out_rf.set_flags(flags);

    // Output the raw log-field for debugging
    std::string random_field_names_log = "log_k";
    data_out_rf.add_data_vector(this->rf_dof_handler,
                                this->x_vec_distributed,
                                random_field_names_log);

    // Also output the exponentiated field k = exp(x_vec)
    // Create an owned-only vector, fill it, then assign to ghosted vector
    TrilinosWrappers::MPI::Vector rf_owned(this->rf_locally_owned, MPI_COMM_WORLD);
    for (const auto i : this->rf_locally_owned)
      rf_owned[i] = std::exp(this->x_vec_distributed[i]);
    rf_owned.compress(VectorOperation::insert);

    TrilinosWrappers::MPI::Vector rf(this->rf_locally_owned,
                                     this->rf_locally_relevant,
                                     MPI_COMM_WORLD);
    rf = rf_owned; // Assignment to ghosted vector updates ghost values


    std::string random_field_names = "k";
    data_out_rf.add_data_vector(this->rf_dof_handler, rf, random_field_names);

    // Use curved cells with subdivision matching the FE degree for nice output.
    // NOTE: In ParaView, keep "Nonlinear Subdivision Level" at 0 to avoid
    // artifacts - ParaView's subdivision algorithm doesn't match FEM
    // interpolation.
    data_out_rf.build_patches(mapping, 1, DataOut<dim>::curved_inner_cells);
    data_out_rf.write_vtu_with_pvtu_record(
      stripped_path_rf, filename_rf, cycle, MPI_COMM_WORLD, n_digits_counter);
  }

  // Main entry point: run the forward Darcy flow simulation.
  // Calls run_simulation and prints timing summary.
  template <int dim>
  void
  DarcyForward<dim>::run(const Parameters &params)
  {
    this->params = params;
    run_simulation();

    this->computing_timer.print_summary();
    this->computing_timer.reset();
    this->pcout << std::endl;
  }

  // Execute complete forward simulation pipeline:
  // setup -> read input -> assemble -> solve -> output results.
  template <int dim>
  void
  DarcyForward<dim>::run_simulation()
  {
    this->setup_grid_and_dofs();
    this->read_input_npy(this->params.input_npy_file);
    // this->generate_ref_input(); // TODO: should be removed in production
    this->generate_coordinates();
    this->assemble_approx_schur_complement();
    this->assemble_system();
    this->solve();

    // Copy solution to ghosted vector for output
    this->solution_distributed = this->solution;
    this->x_vec_distributed    = this->x_vec;

    // output the results
    output_pvtu();
    output_full_velocity_npy();
    output_velocity_at_observation_points_npy();

  } // end run simulation

} // namespace darcy

#endif // DARCY_FORWARD_H
