#include <deal.II/numerics/matrix_tools.h>

#include "darcy.h"
#include "darcy_general.h"

namespace darcy
{

  // ---------------------- read upstream gradient npy -------------------------
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
    int          len_vec  = adjoint_data_vec.size();
    int          num_data = len_vec / dim;
    unsigned int k        = 0;
    data_vec.resize(spatial_coordinates.size(), std::vector<double>(dim));
    for (auto &spatial_coordinate : spatial_coordinates)
      {
        std::vector<double> data_coord(dim * dim); // data_coord(2 * dim);
        for (unsigned int i = 0; i < dim; ++i)
          {
            data_coord[i] = adjoint_data_vec[i * num_data + k];
          }
        data_vec[k] = data_coord;
        ++k;
      }
    pcout << "Successfully read adjoint data!" << std::endl;
    pcout << "Number of adjoint data points: " << data_vec.size() << std::endl;
  }

  // ---------------------- read primary solution npy --------------------------
  template <int dim>
  void
  Darcy<dim>::read_primary_solution(const std::string &output_path)
  {
    TimerOutput::Scope         timing_section(computing_timer,
                                      "read primary solution npy");
    std::vector<unsigned long> shape{};
    bool                       fortran_order;
    std::vector<double>        tmp_primary_solution(
      dof_handler.n_dofs()); // temp vector for primary solution

    // split the input file path into components
    std::string filename = output_path + "_solution_full.npy";
    pcout << "Reading primary solution from " << filename << std::endl;

    // only read the solution on rank 0
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        npy::LoadArrayFromNumpy(filename,
                                shape,
                                fortran_order,
                                tmp_primary_solution);
      }

    double temp;
    // loop over all dofs on distributed solution vector
    for (unsigned int i = 0; i < solution_primary_problem.size(); ++i)
      {
        // copy current dof to double only on rank 0
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          {
            temp = tmp_primary_solution[i];
          }
        // broadcast the value to all processors
        Utilities::MPI::broadcast(MPI_COMM_WORLD, temp, 0);
        // check if current solution dof is on current processor
        if (solution_primary_problem.in_local_range(i))
          {
            solution_primary_problem[i] = temp;
          }
      }
    pcout << "Primary solution read successfully!" << std::endl;
  }

  // ---------------------- overwrite adjoint rhs ------------------------------
  template <int dim>
  void
  Darcy<dim>::overwrite_adjoint_rhs()
  {
    TimerOutput::Scope timing_section(computing_timer, "Overwrite adjoint rhs");
    system_rhs = 0;

    FEValuesExtractors::Vector           velocities(0);
    const unsigned int                   dofs_per_cell = fe.n_dofs_per_cell();
    MappingQ<dim>                        dummy_mapping(1);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // start the cell loop
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        // only consider locally owned cells
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            std::vector<unsigned int> data_element_idx;

            // loop over experimental data points to find points on current cell
            for (unsigned int k = 0; k < data_vec.size(); ++k)
              {
                if (cell->point_inside(spatial_coordinates[k]))
                  {
                    data_element_idx.push_back(k);
                  }
              } // end loop experimental data points on current cell

            // loop over dofs of current cell
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // loop over experimental data points on current cell
                for (unsigned int k = 0; k < data_element_idx.size(); ++k)
                  {
                    unsigned int idx = data_element_idx[k];
                    // transform physical point to unit cell point
                    Point<dim> current_cell_point =
                      dummy_mapping.transform_real_to_unit_cell(
                        cell, spatial_coordinates[idx]);

                    // get shape function value at current point for current
                    // dof-shape fun
                    double shape_value = fe.shape_value(i, current_cell_point);

                    // small hack to filter the correct component --> for
                    // specific dof all but one component should be 0
                    auto   components   = fe.get_nonzero_components(i);
                    double grad_log_lik = 0.0;

                    for (unsigned int comp = 0; comp < components.size();
                         ++comp)
                      {
                        if (components[comp])
                          {
                            grad_log_lik =
                              data_vec[idx]
                                      [comp]; // simple double for grad_log_lik
                                              // value of interest
                            break; // Assuming only one component is non-zero
                                   // per shape function
                          } // end if statement component
                      } // end loop components

                    // write into global rhs (both components for velocity)
                    system_rhs(local_dof_indices[i]) +=
                      -grad_log_lik * shape_value;

                  } // end experimental data loop
              } // end dof loop

          } // end if locally owned
      } // end cell loop

    pcout << "Successfully overwritten rhs..." << std::endl;
  }

  // ---------------------- run method adjoint --------------------------------
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

  // ---------------------- create rf laplace operator -------------------------
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

    // initialize mean vector
    mean_rf.reinit(locally_owned, MPI_COMM_WORLD);


    // quadrature
    const QGauss<dim> quadrature(degree_u + 1); // fine quadrature

    // setup laplace matrix
    // Tell the template exactly which Function type we mean (coefficient == 1)
    const dealii::Function<dim, double> *coefficient = nullptr;
    MatrixCreator::create_laplace_matrix(rf_dof_handler,
                                         quadrature,
                                         rf_laplace_matrix,
                                         coefficient,
                                         rf_constraints);
  }

  // ---------------------- add prior gradient to adjoint ----------------------
  template <int dim>
  void
  Darcy<dim>::add_prior_gradient_to_adjoint()
  {
    // define random field mean vector
    // TODO: adjust mean to BCs

    // apply constraints to laplace matrix and make negative
    // TODO: Dirichlet BCs

    // define a
    const double a = 1e-9 + x_vec.size() / 2.0;

    // initialize vectors
    const IndexSet               &owned = rf_dof_handler.locally_owned_dofs();
    TrilinosWrappers::MPI::Vector x_minus_mean, prior_grad;
    x_minus_mean.reinit(owned, MPI_COMM_WORLD);
    prior_grad.reinit(owned, MPI_COMM_WORLD);

    // Fill x_minus_mean from serial x_vec at owned indices
    for (const auto idx : owned)
      x_minus_mean[idx] = x_vec[idx];
    // x_minus_mean.compress(VectorOperation::insert);

    // Subtract mean directly (mean_rf is already a distributed owned-only
    // vector)
    x_minus_mean.add(-1.0, mean_rf); // x_minus_mean = x_vec - mean_rf


    // b = ||x_vec - mean||^2 + 1e-9
    // double b = x_minus_mean.dot(x_minus_mean) + 1e-9;
    const double nrm_sqr = x_minus_mean.norm_sqr(); // global MPI norm
    const double b       = nrm_sqr + 1e-9;

    // prior_grad = (-b/a) * (L * (x_vec - mean))
    rf_laplace_matrix.vmult(prior_grad, x_minus_mean);
    prior_grad *= (-b / a);


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

  // ---------------------- run simulation adjoint ----------------------------
  template <int dim>
  void
  Darcy<dim>::run_simulation(const std::string &input_path,
                             const std::string &output_path)
  {
    setup_grid_and_dofs();
    read_input_npy(input_path);
    generate_ref_input(); // TODO: should be removed in production
    generate_coordinates();
    read_upstream_gradient_npy(input_path);
    read_primary_solution(output_path); // this needs the dof handler hence
                                        // after setup_grid_and_dofs
    assemble_approx_schur_complement();
    assemble_system(); // we assemble a wrong rhs for the adjoint first and
                       // then overwrite it...
    overwrite_adjoint_rhs();
    solve();
    final_inner_adjoint_product();
    create_rf_laplace_operator();
    add_prior_gradient_to_adjoint();

    // --- write out the gradient with processor 0
    unsigned int rows    = grad_log_lik_x.size();
    unsigned int columns = 1;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string filename = output_path + "_grad_solution.npy";
        write_data_to_npy(filename, grad_log_lik_x, rows, columns);
      }
  }

  // ---------------------- final inner adjoint product -----------------------
  template <int dim>
  void
  Darcy<dim>::final_inner_adjoint_product()
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "Final inner adjoint product");
    pcout << "Final inner adjoint product with jacobi_k_mat_inv" << std::endl;

    // get tensor function and evaluate it at all dofs
    unsigned int   x_dim = x_vec.size();
    Tensor<2, dim> jacobi_k_mat_inv_value;

    // reinit the final gradient vector
    grad_log_lik_x.resize(x_dim);
    grad_log_lik_x_distributed.resize(x_dim);

    // quadrature formula, fe values and dofs
    // standard gauss quadrature
    const QGauss<dim>  quadrature(degree_u +
                                 2); // we choose a coarser quadrature here
    FEValues<dim>      fe_values(fe,
                            quadrature,
                            update_quadrature_points | update_values |
                              update_JxW_values);
    FEValues<dim>      fe_rf_values(rf_fe_system,
                               quadrature,
                               update_values | update_quadrature_points);
    const unsigned int n_q_points = quadrature.size();
    // other stuff
    FEValuesExtractors::Vector velocities(0);
    const unsigned int         dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int rf_dofs_per_cell      = rf_fe_system.n_dofs_per_cell();

    // local to global mapping for random field
    std::vector<types::global_dof_index> local_rf_dof_indices(rf_dofs_per_cell);

    // local to global dof mapping for solution
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    solution_distributed         = solution;
    solution_primary_distributed = solution_primary_problem;

    // instantiate some variables
    Tensor<1, dim>          velocity_dofs_vec;
    std::vector<Point<dim>> q_points(n_q_points);
    double                  JxW_q;

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
            q_points = fe_values.get_quadrature_points();
            std::vector<double> solution_local(dofs_per_cell);
            std::vector<double> solution_primary_local(dofs_per_cell);
            unsigned int        global_dof_index;
            rf_cell->get_dof_indices(local_rf_dof_indices);

            // calculate k-th jacobi matrix
            std::vector<double> rf_value(n_q_points);
            fe_rf_values.get_function_values(x_vec, rf_value);

            // get local solutions once before the quadrature loop
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                global_dof_index  = local_dof_indices[i];
                solution_local[i] = solution_distributed[global_dof_index];
                solution_primary_local[i] =
                  solution_primary_distributed[global_dof_index];
              }
            // ------ loop over quadrature points -------------
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                JxW_q = fe_values.JxW(q);

                // ------- loop over random field dofs per cell ------
                for (unsigned int k = 0; k < rf_dofs_per_cell; ++k)
                  {
                    RandomMedium::get_jacobi_inv_kmat(
                      rf_value[q],
                      fe_rf_values.shape_value(k, q),
                      jacobi_k_mat_inv_value);


                    // ------ outer loop over all dofs of the cell -----------
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const Tensor<1, dim> phi_i_u =
                          fe_values[velocities].value(i, q);
                        double local_solution_i = solution_local[i];

                        // ----- inner loop over all dofs of the cell ---------
                        for (unsigned int j = 0; j < dofs_per_cell; ++j)
                          {
                            const Tensor<1, dim> phi_j_u =
                              fe_values[velocities].value(j, q);
                            double local_primary_j = solution_primary_local[j];


                            // compute inner product and add to global vector
                            grad_log_lik_x_distributed
                              [local_rf_dof_indices[k]] +=
                              local_solution_i *
                              (phi_i_u * jacobi_k_mat_inv_value * phi_j_u *
                               JxW_q) *
                              local_primary_j;
                          } // inner end loop dofs per cell

                      } // end outer dof loop
                  } // end loop over random field dofs

              } // end loop quadrature points

          } // end if cell is locally owned
      } // end loop cell
    pcout << "grad_log_x (distributed) successfully assembled!" << std::endl;

    // sum the proc-wise grad_log_lik_x over all processors to get the final
    // gradient
    // Utilities::MPI::sum(grad_log_lik_x_distributed,
    //                     MPI_COMM_WORLD,
    //                     grad_log_lik_x);
    // pcout << "Successfully summed grad_log_lik_x over processors" <<
    // std::endl;
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
      Darcy<3>                         mixed_laplace_problem(fe_degree);
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
