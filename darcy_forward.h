// darcy_forward.h
// Forward Darcy flow solver class.
// Implements the forward simulation for the Darcy flow problem.

#ifndef DARCY_FORWARD_H
#define DARCY_FORWARD_H

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/mapping_q.h>

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
    explicit DarcyForward(const unsigned int degree_p,
                          const unsigned int degree_rf);

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
    void
    output_spatial_coordinates_npy(); // Observation point coordinates to .npy

    // -------------------------------------------------------------------------
    // Simulation driver
    // -------------------------------------------------------------------------
    void
    run_simulation();
  };

  // Constructor implementation
  template <int dim>
  DarcyForward<dim>::DarcyForward(const unsigned int degree_p,
                                  const unsigned int degree_rf)
    : DarcyBase<dim>(degree_p, degree_rf)
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

    const MappingQ<dim> mapping(2);
    const double        tolerance = 1e-10;
    const unsigned int  n_points  = this->spatial_coordinates.size();

    this->solution_distributed = this->solution;

    // Step 1: Find which cell contains each point (with globally unique
    // assignment)
    std::vector<types::global_cell_index> point_to_cell_id(
      n_points, std::numeric_limits<types::global_cell_index>::max());
    std::vector<typename DoFHandler<dim>::active_cell_iterator> point_to_cell(
      n_points);
    std::vector<Point<dim>> point_to_unit(n_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const types::global_cell_index cell_id =
          cell->global_active_cell_index();

        // Extended bounding box for curved cells (quadratic mapping can
        // cause cell geometry to extend significantly beyond vertex positions)
        BoundingBox<dim> bbox(cell->bounding_box());
        auto             bounds = bbox.get_boundary_points();
        for (unsigned int d = 0; d < dim; ++d)
          {
            const double extent = bounds.second[d] - bounds.first[d];
            bounds.first[d] -= 0.5 * extent;
            bounds.second[d] += 0.5 * extent;
          }
        bbox = BoundingBox<dim>(bounds);

        for (unsigned int k = 0; k < n_points; ++k)
          {
            const Point<dim> &point = this->spatial_coordinates[k];
            if (!bbox.point_inside(point))
              continue;

            try
              {
                const Point<dim> unit_point =
                  mapping.transform_real_to_unit_cell(cell, point);

                if (GeometryInfo<dim>::is_inside_unit_cell(unit_point,
                                                           tolerance))
                  {
                    if (cell_id < point_to_cell_id[k])
                      {
                        point_to_cell_id[k] = cell_id;
                        point_to_cell[k]    = cell;
                        point_to_unit[k]    = unit_point;
                      }
                  }
              }
            catch (...)
              {}
          }
      }

    // Step 2: Global reduction to ensure unique assignment
    std::vector<types::global_cell_index> global_min_cell_id(n_points);
    MPI_Allreduce(point_to_cell_id.data(),
                  global_min_cell_id.data(),
                  n_points,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN,
                  MPI_COMM_WORLD);

    // Step 3: Evaluate solution at points owned by this rank
    std::vector<double> local_values(dim * n_points, 0.0);

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (unsigned int k = 0; k < n_points; ++k)
      {
        if (point_to_cell_id[k] != global_min_cell_id[k] ||
            point_to_cell_id[k] ==
              std::numeric_limits<types::global_cell_index>::max())
          continue;

        const auto       &cell       = point_to_cell[k];
        const Point<dim> &unit_point = point_to_unit[k];

        cell->get_dof_indices(local_dof_indices);

        // Evaluate each velocity component
        for (unsigned int d = 0; d < dim; ++d)
          {
            double value = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int comp =
                  this->fe.system_to_component_index(i).first;
                if (comp == d)
                  {
                    value += this->fe.shape_value(i, unit_point) *
                             this->solution_distributed(local_dof_indices[i]);
                  }
              }
            local_values[d * n_points + k] = value;
          }
      }

    // Step 4: Reduce to rank 0
    std::vector<double> global_values(dim * n_points, 0.0);
    MPI_Reduce(local_values.data(),
               global_values.data(),
               dim * n_points,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    // Step 5: Write on rank 0
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string filename = this->params.output_directory + "/" +
                                     this->params.output_prefix + "sol.npy";
        this->write_data_to_npy(filename,
                                global_values,
                                global_values.size(),
                                1);
      }
  }

  // Write observation point coordinates to "_coordinates.npy".
  // Output format: [all x_1, all x_2, all x_3] - coordinate components stacked.
  // Only rank 0 writes the file.
  template <int dim>
  void
  DarcyForward<dim>::output_spatial_coordinates_npy()
  {
    // Only rank 0 writes the output file
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const unsigned int  n_points = this->spatial_coordinates.size();
        std::vector<double> output_data(dim * n_points);

        // Reorder: [all x_1, all x_2, all x_3]
        for (unsigned int d = 0; d < dim; ++d)
          {
            for (unsigned int i = 0; i < n_points; ++i)
              {
                output_data[d * n_points + i] = this->spatial_coordinates[i][d];
              }
          }

        const std::string filename = this->params.output_directory + "/" +
                                     this->params.output_prefix +
                                     "coordinates.npy";
        this->pcout << "Writing observation coordinates to: " << filename
                    << std::endl;
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
        const std::string file_path = this->params.output_directory + "/" +
                                      this->params.output_prefix +
                                      "solution_full.npy";
        this->pcout << "Writing full solution to file: " << file_path
                    << std::endl;
        this->pcout << "Size of full solution: " << full_solution.size()
                    << std::endl;
        this->pcout << "Number of dofs: " << n_dofs << std::endl;
        this->write_data_to_npy(file_path,
                                full_solution,
                                full_solution.size(),
                                1);
      }
  }

  // Write solution and random field to PVTU files for ParaView visualization.
  // Outputs: "_solution.pvtu" with velocity/pressure, "_random_field.pvtu" with
  // log_k and k.
  template <int dim>
  void
  DarcyForward<dim>::output_pvtu() const
  {
    const std::string filename      = this->params.output_prefix + "solution";
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
    data_out.build_patches(mapping,
                           this->degree_u,
                           DataOut<dim>::curved_inner_cells);

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
    TrilinosWrappers::MPI::Vector rf_owned(this->rf_locally_owned,
                                           MPI_COMM_WORLD);
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
    data_out_rf.build_patches(mapping, 2, DataOut<dim>::curved_inner_cells);
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
    this->read_input_npy();
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
    output_spatial_coordinates_npy();

  } // end run simulation

} // namespace darcy

#endif // DARCY_FORWARD_H
