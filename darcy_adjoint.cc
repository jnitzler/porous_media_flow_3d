#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/matrix_tools.h>

#include "darcy.h"
#include "darcy_general.h"

namespace darcy
{
  // Read upstream gradient (dL/dy) from "adjoint_data.npy".
  // Reshapes flat array [u1_all, u2_all, ...] into data_vec[point][component].
  template <int dim>
  void
  Darcy<dim>::read_upstream_gradient_npy(const std::string &input_file_path)
  {
    TimerOutput::Scope timing_section(computing_timer,
                                      "read upstream gradient npy");
    // split the input file path into components
    std::filesystem::path my_path(input_file_path);
    std::filesystem::path base_dir = my_path.parent_path();

    const std::string adjoint_file_path =
      base_dir.string() + "/adjoint_data.npy";
    pcout << "Reading adjoint data from: " << adjoint_file_path << std::endl;

    std::vector<unsigned long> shape{};
    bool                       fortran_order;

    npy::LoadArrayFromNumpy(adjoint_file_path,
                            shape,
                            fortran_order,
                            adjoint_data_vec);

    // NOTE: the adjoint data vec has the following organization: velocity_1
    // block, velocity_2, etc block construct final data vector from range based
    // loop over time points

    // stucture of one data_vec: [grad_log_lik_y1, grad_log_lik_y2,
    // grad_log_lik_y3]
    unsigned int len_vec      = adjoint_data_vec.size();
    unsigned int num_data     = len_vec / (dim);
    unsigned int spatial_size = spatial_coordinates.size();
    data_vec.resize(spatial_size, std::vector<double>(dim));

    AssertThrow(num_data == spatial_size,
                ExcMessage("Mismatch: num_data=" + std::to_string(num_data) +
                           " vs spatial_size=" + std::to_string(spatial_size)));

    for (unsigned int k = 0; k < spatial_size; ++k)
      {
        std::vector<double> data_coord(dim);
        for (unsigned int i = 0; i < dim; ++i)
          {
            data_coord[i] = adjoint_data_vec[i * num_data + k];
          }
        data_vec[k] = data_coord;
      }
  }

  // Load forward solution from "_solution_full.npy" into
  // solution_primary_problem. Required for computing adjoint inner product.
  template <int dim>
  void
  Darcy<dim>::read_primary_solution(const std::string &output_path)
  {
    dealii::TimerOutput::Scope timing_section(computing_timer,
                                              "read primary solution npy");
    std::vector<unsigned long> shape{};
    bool                       fortran_order;

    // split the input file path into components
    std::string file_path = output_path + "_solution_full.npy";
    pcout << "Reading primary solution from " << file_path << std::endl;

    std::vector<double> tmp_primary_solution;
    tmp_primary_solution.resize(
      dof_handler.n_dofs()); // temp vector for primary solution

    npy::LoadArrayFromNumpy(file_path,
                            shape,
                            fortran_order,
                            tmp_primary_solution);

    pcout << "Primary solution read successfully!" << std::endl;
    // loop over all dofs on distributed solution vector
    pcout << "Writing primary solution to distributed solution vector..."
          << std::endl;

    for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
      {
        if (solution_primary_problem.in_local_range(i))
          {
            solution_primary_problem[i] = tmp_primary_solution[i];
          }
      }
    solution_primary_problem.compress(VectorOperation::insert);
    pcout
      << "Primary solution successfully written to distributed solution vector"
      << std::endl;
  }

  // Replace system_rhs with adjoint source term: sum_k (dL/dy_k) * phi_i(x_k).
  // Evaluates shape functions at observation points and weights by upstream
  // gradient.
  template <int dim>
  void
  Darcy<dim>::overwrite_adjoint_rhs()
  {
    TimerOutput::Scope timing_section(computing_timer, "Overwrite adjoint rhs");

    system_rhs                                         = 0;
    const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
    MappingQ<dim>                        dummy_mapping(2);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // start the cell loop
    for (const auto &cell_tria : triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(dof_handler);
        // only consider locally owned cells
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            std::vector<unsigned int> data_element_idx;

            // loop over experimental data points to find points on current cell
            for (unsigned int k = 0; k < spatial_coordinates.size(); ++k)
              {
                if (cell->point_inside(spatial_coordinates[k]))
                  {
                    // stores indices of data points that fall into current cell
                    data_element_idx.push_back(k);
                  }
              } // end loop experimental data points on current cell

            // loop over data points that fall into current cell
            for (unsigned int k = 0; k < data_element_idx.size(); ++k)
              {
                // transform physical point to unit cell point
                Point<dim> current_cell_point =
                  dummy_mapping.transform_real_to_unit_cell(
                    cell, spatial_coordinates[data_element_idx[k]]);

                // loop over dofs of current cell
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // Get the component index for this DOF
                    // system_to_component_index returns a pair:
                    //   .first  = component (0, 1, ..., dim-1 for velocity, dim
                    //   for pressure) .second = index within that component's
                    //   scalar element
                    const unsigned int comp =
                      fe.system_to_component_index(i).first;

                    // Skip pressure DOFs (component == dim)
                    if (comp >= dim)
                      continue;

                    double shape_value  = fe.shape_value(i, current_cell_point);
                    double grad_log_lik = data_vec[data_element_idx[k]][comp];

                    // write into global rhs
                    system_rhs(local_dof_indices[i]) +=
                      grad_log_lik * shape_value;

                  } // end dof loop
              } // end experimental data loop

          } // end if locally owned
      } // end cell loop
    system_rhs.compress(VectorOperation::add);
    pcout << "Successfully overwritten rhs..." << std::endl;
    pcout << "Adjoint rhs norm: " << system_rhs.l2_norm() << std::endl;
  }

  // Main entry point for adjoint computation.
  // Runs full adjoint pipeline and prints timing summary.
  template <int dim>
  void
  Darcy<dim>::run(const std::string &input_path, const std::string &output_path)
  {
    run_simulation(input_path, output_path);

    pcout << "Adjoint problem solved successfully!" << std::endl;
    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
  }

  // Build Laplace matrix L for GMRF prior on random field.
  // Adds small mass matrix (nugget) for positive definiteness: Q = L + eps*M.
  template <int dim>
  void
  Darcy<dim>::create_rf_laplace_operator()
  {
    TimerOutput::Scope timing_section(computing_timer,
                                      "Create rf laplace operator");
    pcout << "Creating random field laplace operator..." << std::endl;

    //  index sets
    const IndexSet locally_owned = rf_dof_handler.locally_owned_dofs();
    IndexSet       locally_relevant;
    DoFTools::extract_locally_relevant_dofs(rf_dof_handler, locally_relevant);

    // constraints
    AffineConstraints<double> rf_constraints;
    rf_constraints.clear();
    DoFTools::make_hanging_node_constraints(rf_dof_handler, rf_constraints);
    rf_constraints.close();

    // Trilinos sparsity pattern
    TrilinosWrappers::SparsityPattern sp_rf(locally_owned,
                                            locally_owned,
                                            locally_relevant,
                                            MPI_COMM_WORLD);

    DoFTools::make_sparsity_pattern(rf_dof_handler,
                                    sp_rf,
                                    rf_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    sp_rf.compress();

    // initialize the matrix

    rf_laplace_matrix.reinit(sp_rf);

    // initialize mean vector and set to zero (prior mean = 0)
    mean_rf.reinit(locally_owned, MPI_COMM_WORLD);
    mean_rf = 0;


    // quadrature
    const QGauss<dim> quadrature(degree_u + 1); // fine quadrature

    // Use higher-order mapping for curved geometry
    MappingQ<dim> mapping(2);

    // setup laplace matrix
    // Tell the template exactly which Function type we mean (coefficient == 1)
    const dealii::Function<dim, double> *coefficient = nullptr;
    MatrixCreator::create_laplace_matrix(mapping,
                                         rf_dof_handler,
                                         quadrature,
                                         rf_laplace_matrix,
                                         coefficient,
                                         rf_constraints);
    // 2. Assemble the Mass Matrix (nugget)
    TrilinosWrappers::SparseMatrix rf_mass_matrix(sp_rf);

    dealii::MatrixCreator::create_mass_matrix(mapping,
                                              rf_dof_handler,
                                              quadrature,
                                              rf_mass_matrix,
                                              coefficient,
                                              rf_constraints);

    // 3. Combine them: Q = L + epsilon * M
    // A tiny epsilon (e.g., 1e-8) is enough to make it positive definite
    // without noticeably changing the correlation structure.
    const double epsilon = 1e-5;
    rf_laplace_matrix.add(epsilon, rf_mass_matrix);
  }

  // Add GMRF prior gradient: grad_log_prior = -(a/b) * L * (x - mu).
  // Uses Gamma hyperprior on precision with empirical Bayes update.
  template <int dim>
  void
  Darcy<dim>::add_prior_gradient_to_adjoint()
  {
    // define a (use global number of DOFs, not local size)
    const double a = 1e-9 + rf_dof_handler.n_dofs() / 2.0;

    // initialize vectors
    const IndexSet               &owned = rf_dof_handler.locally_owned_dofs();
    TrilinosWrappers::MPI::Vector x_minus_mean, prior_grad;
    x_minus_mean.reinit(owned, MPI_COMM_WORLD);
    prior_grad.reinit(owned, MPI_COMM_WORLD);

    // Fill x_minus_mean from distributed x_vec at owned indices
    // x_vec already has the correct values for owned DOFs
    for (const auto idx : owned)
      x_minus_mean[idx] = x_vec[idx];
    x_minus_mean.compress(VectorOperation::insert);

    // Subtract mean directly (mean_rf is already a distributed owned-only
    // vector)
    x_minus_mean.add(-1.0, mean_rf); // x_minus_mean = x_vec - mean_rf

    // Compute L(x - μ)
    rf_laplace_matrix.vmult(prior_grad,
                            x_minus_mean); // prior_grad = L * (x - μ)

    // Compute quadratic form (x-μ)ᵀ L (x-μ) for b
    const double b_0       = 1e-9;
    const double quad_form = x_minus_mean * prior_grad; // (x-μ)ᵀ L (x-μ)
    const double b         = b_0 + 0.5 * quad_form;

    // Scale: prior_grad = -(a/b) * L(x - μ)
    prior_grad *= -(a / b);


    // add prior gradient to adjoint gradient
    for (const auto idx : owned)
      grad_log_lik_x_distributed[idx] += prior_grad[idx];

    // Reduce to a global (serial) vector for output
    Utilities::MPI::sum(grad_log_lik_x_distributed,
                        MPI_COMM_WORLD,
                        grad_log_lik_x);

    pcout << "Successfully added prior gradient to adjoint gradient!"
          << std::endl;
  }

  // Write gradient field to "_gradient.pvtu" for visualization.
  template <int dim>
  void
  Darcy<dim>::output_gradient_vtu(const std::string &output_path)
  {
    TimerOutput::Scope timing_section(computing_timer, "Output gradient VTU");
    pcout << "Writing gradient to VTU file..." << std::endl;

    // Create a distributed vector for the gradient on the rf_dof_handler
    const IndexSet &locally_owned = rf_dof_handler.locally_owned_dofs();
    IndexSet        locally_relevant;
    DoFTools::extract_locally_relevant_dofs(rf_dof_handler, locally_relevant);

    // Owned-only vector
    TrilinosWrappers::MPI::Vector gradient_owned(locally_owned, MPI_COMM_WORLD);

    // Distributed vector with ghost values for output
    TrilinosWrappers::MPI::Vector gradient_distributed(locally_owned,
                                                       locally_relevant,
                                                       MPI_COMM_WORLD);

    // Fill the gradient from the std::vector (which has global values after
    // MPI::sum)
    for (const auto idx : locally_owned)
      gradient_owned[idx] = grad_log_lik_x[idx];

    // Copy to distributed vector (updates ghosts)
    gradient_distributed = gradient_owned;

    // Setup DataOut
    DataOut<dim> data_out;
    data_out.attach_dof_handler(rf_dof_handler);
    data_out.add_data_vector(gradient_distributed, "gradient");

    MappingQ<dim> mapping(2);
    data_out.build_patches(mapping, 2, DataOut<dim>::curved_inner_cells);

    // Write output - extract directory and filename from output_path
    const std::size_t found    = output_path.find_last_of("/\\");
    const std::string filename = output_path.substr(found + 1) + "_gradient";
    const std::string stripped_path = output_path.substr(0, found + 1);

    data_out.write_vtu_with_pvtu_record(
      stripped_path, filename, 0, MPI_COMM_WORLD, 1, 1);

    pcout << "Gradient written to " << stripped_path << filename << ".pvtu"
          << std::endl;
  }

  // Execute complete adjoint pipeline:
  // setup -> read data -> solve adjoint -> compute gradient -> add prior ->
  // output.
  template <int dim>
  void
  Darcy<dim>::run_simulation(const std::string &input_path,
                             const std::string &output_path)
  {
    const bool adjoint_solve = true;

    setup_grid_and_dofs();
    read_input_npy(input_path);
    // generate_ref_input(); // TODO: should be removed in production
    generate_coordinates();
    read_upstream_gradient_npy(input_path);
    read_primary_solution(output_path);
    assemble_approx_schur_complement();
    assemble_system();
    overwrite_adjoint_rhs();
    solve(adjoint_solve);
    write_adjoint_solution_pvtu(output_path);
    final_inner_adjoint_product();
    create_rf_laplace_operator();
    add_prior_gradient_to_adjoint();
    output_gradient_vtu(output_path);
    write_gradient_to_npy(output_path);
  }

  // Write gradient to "_grad_solution.npy". Only rank 0 writes.
  template <int dim>
  void
  Darcy<dim>::write_gradient_to_npy(const std::string &output_path)
  {
    // --- write out the gradient with processor 0
    unsigned int rows    = grad_log_lik_x.size();
    unsigned int columns = 1;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string filename = output_path + "_grad_solution.npy";
        write_data_to_npy(filename, grad_log_lik_x, rows, columns);
      }
  }

  // Write adjoint solution (velocity, pressure) to "_adjoint_solution.pvtu".
  template <int dim>
  void
  Darcy<dim>::write_adjoint_solution_pvtu(const std::string &output_path)
  {
    // Copy solution to distributed vector with ghost values
    solution_distributed = solution;

    // Create component names for velocity and pressure
    std::vector<std::string> solution_names(dim, "adjoint_velocity");
    solution_names.emplace_back("adjoint_pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.emplace_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<dim>          data_out;
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.add_data_vector(dof_handler,
                             solution_distributed,
                             solution_names,
                             interpretation);

    // build the patches - same as darcy.cc
    MappingQ<dim> mapping(2); // nonlinear mapping
    data_out.build_patches(mapping, degree_u, DataOut<dim>::curved_inner_cells);

    // Extract directory and filename from output_path
    const std::size_t found = output_path.find_last_of("/\\");
    const std::string filename =
      output_path.substr(found + 1) + "_adjoint_solution";
    const std::string stripped_path = output_path.substr(0, found + 1);

    constexpr unsigned int n_digits_counter = 2;
    constexpr unsigned int num_vtu_files    = 1;
    constexpr unsigned int cycle            = 0;
    data_out.write_vtu_with_pvtu_record(stripped_path,
                                        filename,
                                        cycle,
                                        MPI_COMM_WORLD,
                                        n_digits_counter,
                                        num_vtu_files);

    pcout << "DEBUG: Wrote adjoint solution to " << stripped_path << filename
          << ".pvtu" << std::endl;
    pcout << "DEBUG: adjoint solution size = " << dof_handler.n_dofs()
          << ", rf_dof_handler.n_dofs() = " << rf_dof_handler.n_dofs()
          << std::endl;
    pcout << "DEBUG: adjoint solution norm = " << solution.l2_norm()
          << std::endl;
  }

  // Compute gradient w.r.t. random field: dL/dx = -lambda^T (dA/dx) u.
  // Evaluates adjoint-primary velocity inner product weighted by d(K^{-1})/dx.
  template <int dim>
  void
  Darcy<dim>::final_inner_adjoint_product()
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "Final inner adjoint product");
    pcout << "Final inner adjoint product with jacobi_k_mat_inv" << std::endl;

    // get tensor function and evaluate it at all dofs
    unsigned int x_dim = rf_dof_handler.n_dofs();

    // reinit the final gradient vector - MUST initialize to zero!
    grad_log_lik_x.assign(x_dim, 0.0);
    grad_log_lik_x_distributed.assign(x_dim, 0.0);

    const MappingQ<dim> mapping(2);

    // quadrature formula, fe values and dofs
    const QGauss<dim> quadrature(degree_u + 1);

    // FEValues with proper mapping for curved geometry
    FEValues<dim>      fe_values(mapping,
                            fe,
                            quadrature,
                            update_values | update_JxW_values);
    FEValues<dim>      fe_rf_values(mapping,
                               rf_fe_system,
                               quadrature,
                               update_values);
    const unsigned int n_q_points = fe_values.n_quadrature_points;

    // extractors and sizes
    FEValuesExtractors::Vector velocities(0);
    const unsigned int         dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int rf_dofs_per_cell      = rf_fe_system.n_dofs_per_cell();

    // local to global mapping for random field
    std::vector<types::global_dof_index> local_rf_dof_indices(rf_dofs_per_cell);

    // local to global dof mapping for solution
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // Pre-allocate vectors outside cell loop
    std::vector<double>         solution_local(dofs_per_cell);
    std::vector<double>         solution_primary_local(dofs_per_cell);
    std::vector<double>         rf_value(n_q_points);
    std::vector<Tensor<1, dim>> velocity_adjoint_q(n_q_points);
    std::vector<Tensor<1, dim>> velocity_primary_q(n_q_points);
    std::vector<double>         local_gradient(rf_dofs_per_cell);

    // Copy solutions to distributed vectors with ghosts
    solution_distributed         = solution;
    solution_primary_distributed = solution_primary_problem;

    // start the cell loop
    for (const auto &cell_tria : triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(rf_dof_handler);

        // only consider locally owned cells
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_rf_values.reinit(rf_cell);

            cell->get_dof_indices(local_dof_indices);
            rf_cell->get_dof_indices(local_rf_dof_indices);

            // get random field values at quadrature points
            fe_rf_values.get_function_values(x_vec_distributed, rf_value);

            // get local solutions
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                solution_local[i] = solution_distributed[local_dof_indices[i]];
                solution_primary_local[i] =
                  solution_primary_distributed[local_dof_indices[i]];
              }

            // Compute velocity fields at all quadrature points
            // velocity_adjoint = sum_i solution_local[i] * phi_i_u(q)
            // velocity_primary = sum_j primary_local[j] * phi_j_u(q)
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                velocity_adjoint_q[q] = 0;
                velocity_primary_q[q] = 0;
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const Tensor<1, dim> phi_i_u =
                      fe_values[velocities].value(i, q);
                    velocity_adjoint_q[q] += solution_local[i] * phi_i_u;
                    velocity_primary_q[q] +=
                      solution_primary_local[i] * phi_i_u;
                  }
              }

            // Reset local gradient accumulator
            std::fill(local_gradient.begin(), local_gradient.end(), 0.0);

            // Loop over quadrature points
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const double JxW_q = fe_values.JxW(q);

                // Precompute the scalar factor from the jacobian of K^{-1}:
                // d/dx(K^{-1}) = -exp(-rf) * I * phi_k
                // So: velocity_adjoint · (d/dx K^{-1}) · velocity_primary
                //   = -exp(-rf) * phi_k * (velocity_adjoint · velocity_primary)
                const double exp_neg_rf = 1.0 / std::exp(rf_value[q]);
                const double vel_dot =
                  velocity_adjoint_q[q] * velocity_primary_q[q];
                const double common_factor = JxW_q * exp_neg_rf * vel_dot;

                // Loop over random field dofs per cell - O(rf_dofs_per_cell)
                // instead of O(dofs_per_cell^2)
                for (unsigned int k = 0; k < rf_dofs_per_cell; ++k)
                  {
                    const double phi_k = fe_rf_values.shape_value(k, q);
                    // The negative sign comes from get_jacobi_inv_kmat
                    local_gradient[k] += common_factor * phi_k;
                  }
              } // end loop quadrature points

            // Scatter local gradient to global distributed vector
            for (unsigned int k = 0; k < rf_dofs_per_cell; ++k)
              {
                grad_log_lik_x_distributed[local_rf_dof_indices[k]] +=
                  local_gradient[k];
              }

          } // end if cell is locally owned
      } // end loop cell
    pcout << "grad_log_x (distributed) successfully assembled!" << std::endl;
  }

} // end namespace darcy

// ---------------------- main function adjoint -----------------------------
int
main(int argc, char *argv[])
{
  std::string input_file_path  = argv[1];
  std::string output_file_path = argv[2];

  try
    {
      using namespace darcy;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      const unsigned int               fe_degree = 1;
      Darcy<2>                         mixed_laplace_problem(fe_degree);
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
