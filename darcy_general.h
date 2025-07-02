// darcy_general.h
#ifndef DARCY_GENERAL_H
#define DARCY_GENERAL_H

#include "darcy.h"
#include "preconditioner.h"
#include "random_permeability.h"

namespace darcy
{
  // ---------------- generate reference input from function -----------
  template <int dim>
  void
  Darcy<dim>::generate_ref_input()
  {
    const RandomMedium::RefScalar<dim> ref_scalar;
    x_vec.reinit(rf_dof_handler.n_dofs());
    VectorTools::interpolate(rf_dof_handler, ref_scalar, x_vec);
    x_vec = 1.0;
  }

  // generate coordinates at which observations are present
  template <int dim>
  void
  Darcy<dim>::generate_coordinates()
  {
    // ------ generate coordinates ----------------//
    const unsigned int n_points = 20;
    const double       h        = 1.0 / (n_points - 1);
    Point<dim>         p;
    spatial_coordinates.resize(Utilities::fixed_power<dim>(n_points));

    for (unsigned int idx = 0; idx < spatial_coordinates.size(); ++idx)
      {
        unsigned int tempIdx = idx;
        for (int d = 0; d < dim; ++d)
          {
            p[d] = (tempIdx % n_points) * h;
            tempIdx /= n_points;
          }
        spatial_coordinates[idx] = p;
      }
  }

  // ------ class constructor ----------------
  template <int dim>
  Darcy<dim>::Darcy(const unsigned int degree)
    : degree(degree)
    , triangulation(MPI_COMM_WORLD,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
    , rf_fe_system(FE_Q<dim>(1), 1)
    , rf_dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(MPI_COMM_WORLD,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)

  {}

  // ------ read input file ------------------------------
  template <int dim>
  void
  Darcy<dim>::read_input_npy(const std::string &filename)
  {
    TimerOutput::Scope timer_section(computing_timer, "   Read Inputs");

    std::vector<unsigned long> shape{};
    bool                       fortran_order{};

    // read in the permeability field
    std::vector<double> x_std_vec;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, x_std_vec);

    unsigned int n_dofs_rf = rf_dof_handler.n_dofs();
    x_vec.reinit(n_dofs_rf);
    pcout << "Read in random field from file: " << filename << std::endl;
    pcout << "Number of random field dofs: " << x_vec.size() << std::endl;
    pcout << "Number of input field dofs: " << x_std_vec.size() << std::endl;
    for (unsigned int i = 0; i < n_dofs_rf; ++i)
      {
        x_vec[i] = x_std_vec[i];
      }
    pcout << "Random field successfully read in." << std::endl;
  }


  // -------- pressure boundary values ----------------------------
  template <int dim>
  class PressureBoundaryValues : public Function<dim>
  {
  public:
    PressureBoundaryValues()
      : Function<dim>(1)
    {}

    double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };

  template <int dim>
  double
  PressureBoundaryValues<dim>::value(const Point<dim> &p,
                                     const unsigned int /*component*/) const
  {
    return 0.0;
  }

  // -------- velocity boundary values ----------------------------
  template <int dim>
  class VelocityBoundaryValues : public Function<dim>
  {
  public:
    VelocityBoundaryValues()
      : Function<dim>(3)
    {}

    void
    vector_value(const Point<dim> &p, Vector<double> &vector) const override;
  };

  template <int dim>
  void
  VelocityBoundaryValues<dim>::vector_value(const Point<dim> &p,
                                            Vector<double>   &vector) const
  {
    vector[0] = 0.0;
    vector[1] = 0.0;
    vector[2] = p[2]; // pressure
  }


  // ------------- assemble preconditioner ---------
  template <int dim>
  void
  Darcy<dim>::assemble_approx_schur_complement()
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Assemble preconditioner");
    pcout << "Assemble preconditioner..." << std::endl;
    precondition_matrix = 0;
    const QGauss<dim>     quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 2);

    // start the cell loop
    FEValues<dim>      fe_values(fe,
                            quadrature_formula,
                            update_JxW_values | update_values |
                              update_quadrature_points | update_gradients);
    FEFaceValues<dim>  fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_gradients | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    FEValues<dim>      fe_rf_values(rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    for (const auto &cell_tria : triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(rf_dof_handler);

        // only consider locally owned cells
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            fe_rf_values.reinit(rf_cell);

            cell->get_dof_indices(local_dof_indices);

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);

            local_matrix = 0;

            // get rf function values and permeability tensor per cell
            std::vector<double> rf_values(n_q_points);
            Tensor<2, dim>      K_mat;
            fe_rf_values.get_function_values(x_vec, rf_values);

            // quadrature loop
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // evaluate random field at quadrature point
                RandomMedium::get_k_mat(rf_values[q], K_mat);

                // evaluate fe values on all dofs first
                const auto JxW_q = fe_values.JxW(q);
                std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                  }

                // loop over cell dofs
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    for (unsigned int j = 0; j <= i; ++j)
                      {
                        // assemble local schur matrix
                        local_matrix(i, j) +=
                          (K_mat * grad_phi_p[i] * grad_phi_p[j]) * JxW_q;
                      } // end inner dof loop

                  } // end outer dof loop
              }     // end quadrature loop

            // take care of the symmetries
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                {
                  local_matrix(i, j) = local_matrix(j, i);
                }

            for (const auto &face : cell->face_iterators())
              {
                if (face->at_boundary() && face->boundary_id() == 1)
                  {
                    fe_face_values.reinit(cell, face);

                    // (k+1)^2 / h
                    const auto tau = 4. * 2. * 2. / cell->diameter();

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        const auto normal = fe_face_values.normal_vector(q);

                        // loop over face dofs i
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const double phi_i_p =
                              fe_face_values[pressure].value(i, q);

                            const auto grad_phi_i_p =
                                    fe_face_values[pressure].gradient(i, q);

                            // loop over other face dofs j
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const double phi_j_p =
                                  fe_face_values[pressure].value(j, q);

                                const auto grad_phi_j_p =
                                        fe_face_values[pressure].gradient(j, q);

                                local_matrix(i, j) += (-grad_phi_i_p * normal * phi_j_p
                                  - (phi_i_p * (grad_phi_j_p * normal - tau * phi_j_p))) *
                                  fe_face_values.JxW(q);
                              } // end inner dof loop j
                          }     // end face dof loops i
                      }         // end quadrature loop for faces
                  }             // end if statement
              }                 // end face loop

            preconditioner_constraints.distribute_local_to_global(
              local_matrix, local_dof_indices, precondition_matrix);

          } // end if locally owned

      } // end cell loop

    precondition_matrix.compress(VectorOperation::add);
    pcout << "Preconditioner successfully assembled" << std::endl;
  }

  // ------------- assemble system -----------------
  template <int dim>
  void
  Darcy<dim>::assemble_system()
  {
    TimerOutput::Scope timer_section(computing_timer, "  Assemble system");
    pcout << "Assemble system..." << std::endl;

    system_matrix = 0;
    system_rhs    = 0;

    const QGauss<dim>     quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 2);
    FEValues<dim>         fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEValues<dim>         fe_rf_values(rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    FEFaceValues<dim>     fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    const unsigned int    dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int    n_q_points      = fe_values.n_quadrature_points;
    const unsigned int    n_face_q_points = fe_face_values.n_quadrature_points;
    std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
    std::vector<double>                  div_phi_u(dofs_per_cell);
    std::vector<double>                  phi_p(dofs_per_cell);
    std::vector<Tensor<1, dim>>          grad_phi_p(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

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

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);

            local_matrix = 0;
            local_rhs    = 0;

            // pressure boundary values
            std::vector<double> boundary_values_pressure(n_face_q_points);
            const PressureBoundaryValues<dim> pressure_boundary_values;

            // get rf function values and permeability tensor per cell
            // note we assume the same quadrature points for random field and
            // solution values
            std::vector<double> rf_values(n_q_points);
            Tensor<2, dim>      K_mat;
            fe_rf_values.get_function_values(x_vec, rf_values);

            // ---- INTERIOR loop over quadrature points ------------ //
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // get the permeability tensor at quadrature point
                RandomMedium::get_k_mat(rf_values[q], K_mat);
                const Tensor<2, dim> k_inverse = invert(K_mat);

                // evaluate fe values on all dofs first
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]      = fe_values[velocities].value(k, q);
                    div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                    phi_p[k]      = fe_values[pressure].value(k, q);
                    grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                  }

                // loop over cell dofs
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    // for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        // assemble local system matrix
                        local_matrix(i, j) +=
                          (phi_u[i] * k_inverse * phi_u[j] -
                           phi_p[j] * div_phi_u[i] // old term from
                           // divergence direct
                           + grad_phi_p[i] *
                               phi_u[j] // new term from divergence by parts
                           //+ grad_phi_p[j] *
                           //    phi_u[i] // new term from divergence by parts
                           //- div_phi_u[i] * phi_p[j] // old original div term
                           ) *
                          fe_values.JxW(q);
                      } // end inner dof loop

                    // take care of source term
                    local_rhs(i) += (-phi_p[i] * 1.0) * fe_values.JxW(q); //
                  } // end outer dof loop
              }

            // ------ FACE loops over faces (split in two parts) ---- //
            // part 1: weak pressure BC on outer boundary
            for (const auto &face : cell->face_iterators())
              {
                if (face->at_boundary() && face->boundary_id() == 1)
                  {
                    fe_face_values.reinit(cell, face);

                    pressure_boundary_values.value_list(
                      fe_face_values.get_quadrature_points(),
                      boundary_values_pressure);

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        // loop over face dofs i
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const Tensor<1, dim> phi_i_u =
                              fe_face_values[velocities].value(i, q);

                            const double phi_i_p =
                              fe_face_values[pressure].value(i, q);

                            // add contribution to local rhs of the pressure
                            // (weakly)
                            local_rhs(i) += -(phi_i_u * //
                                              fe_face_values.normal_vector(q) *
                                              boundary_values_pressure[q] *
                                              fe_face_values.JxW(q));

                            // loop over other face dofs j
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const Tensor<1, dim> phi_j_u =
                                  fe_face_values[velocities].value(j, q);
                                // const double phi_j_p =
                                //   fe_face_values[pressure].value(j, q);

                                // add contribution to local matrix from unknown
                                // velocity
                                local_matrix(i, j) -=
                                  (phi_i_p * (fe_face_values.normal_vector(q) *
                                              phi_j_u)) * // +
                                  // phi_j_p * (fe_face_values.normal_vector(q)
                                  // *
                                  //           phi_i_u)) *
                                  fe_face_values.JxW(q);
                              } // end inner dof loop j
                          }     // end face dof loops i
                      }         // end quadrature loop for faces
                  }             // end if statement

                // part 2: weak velocity BC on inner boundary
                if (face->at_boundary() && face->boundary_id() == 0)
                  {
                    fe_face_values.reinit(cell, face);

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            // add contribution to local rhs of the velocity
                            // (weakly) -> no contribution as u is zero on
                            // this boundary
                            const Tensor<1, dim> phi_i_u =
                              fe_face_values[velocities].value(i, q);


                            // add contribution to local matrix of the
                            // pressure
                            // loop over other face dofs j
                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const double phi_j_p =
                                  fe_face_values[pressure].value(j, q);


                                local_matrix(i, j) +=
                                  (phi_j_p * //
                                   (phi_i_u * fe_face_values.normal_vector(q)) *
                                   fe_face_values.JxW(q));

                              } // end inner dof loop j
                          }     // end face dof loops i
                      }         // end quadrature loop for faces
                  }             // end if statement
              }                 // end face loop

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);

          } // end if locally owned
      }     // end cell loop

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    pcout << "System successfully assembled" << std::endl;
  }

  // ---------- setup system matrix ------------------------
  template <int dim>
  void
  Darcy<dim>::setup_system_matrix(
    const std::vector<IndexSet> &partitioning,
    const std::vector<IndexSet> &relevant_partitioning)
  {
    system_matrix.clear();
    block_mass_matrix.clear();

    TrilinosWrappers::BlockSparsityPattern sp(partitioning,
                                              partitioning,
                                              relevant_partitioning,
                                              MPI_COMM_WORLD);

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling,
                                    sp,
                                    constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    sp.compress();

    system_matrix.reinit(sp);
    block_mass_matrix.reinit(
      sp); // we just use the same sparsity pattern
           // for the block mass matrix (needed for adjoint) here
           // this should be moved to a separate function in the future
  }

  // ---------- setup preconditioner ------------------------
  template <int dim>
  void
  Darcy<dim>::setup_approx_schur_complement(
    const std::vector<IndexSet> &partitioning,
    const std::vector<IndexSet> &relevant_partitioning)
  {
    precondition_matrix.clear();

    TrilinosWrappers::BlockSparsityPattern sp(partitioning,
                                              partitioning,
                                              relevant_partitioning,
                                              MPI_COMM_WORLD);

    Table<2, DoFTools::Coupling> coupling_precond(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (c == d)
          coupling_precond[c][d] = DoFTools::always;
        else
          coupling_precond[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(dof_handler,
                                    coupling_precond,
                                    sp,
                                    constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    sp.compress();

    precondition_matrix.reinit(sp);
  }

  // --------- setup grid and dofs ----------------
  template <int dim>
  void
  Darcy<dim>::setup_grid_and_dofs()
  {
    TimerOutput::Scope timing_section(computing_timer, "Setup dof systems");
    // velocity components are block 0 and pressure components are block 1
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;

    // generate grid and distribute dofs
    Point<dim>   inner_center(0.0, 0.1, 0.25);
    Point<dim>   outer_center(0.0, 0.0, 0.0);
    double       inner_radius = 0.3; // inner radius
    double       outer_radius = 1.0; // outer radius
    unsigned int n_cells      = 12;  // n_cells

    GridGenerator::eccentric_hyper_shell(triangulation,
                                         inner_center,
                                         outer_center,
                                         inner_radius,
                                         outer_radius,
                                         n_cells);

    triangulation.refine_global(4);
    dof_handler.distribute_dofs(fe);

    // generate grid and distribute dofs for random field
    rf_dof_handler.distribute_dofs(rf_fe_system);
    DoFRenumbering::Cuthill_McKee(
      rf_dof_handler); // Cuthill_McKee, component_wise to be more efficient

    // component wise renumbering
    DoFRenumbering::Cuthill_McKee(
      dof_handler); // Cuthill_McKee, component_wise to be more efficient
    DoFRenumbering::component_wise(dof_handler, block_component);

    // count dofs per block
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

    const types::global_dof_index n_u = dofs_per_block[0],
                                  n_p = dofs_per_block[1];

    std::locale s = pcout.get_stream().getloc();
    pcout.get_stream().imbue(std::locale(""));
    pcout << "Number of active cells: " << triangulation.n_active_cells()
          << std::endl
          << "Total number of cells: " << triangulation.n_cells() << std::endl
          << "Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
          << n_u << '+' << n_p << ')' << std::endl
          << "Number of random field dofs: " << rf_dof_handler.n_dofs()
          << std::endl;
    pcout.get_stream().imbue(s);

    // create relevant index set
    std::vector<IndexSet> partitioning, relevant_partitioning;
    IndexSet              relevant_set;
    {
      IndexSet index_set = dof_handler.locally_owned_dofs();
      partitioning.push_back(index_set.get_view(0, n_u));
      partitioning.push_back(index_set.get_view(n_u, n_u + n_p));

      relevant_set = DoFTools::extract_locally_relevant_dofs(dof_handler);
      relevant_partitioning.push_back(relevant_set.get_view(0, n_u));
      relevant_partitioning.push_back(relevant_set.get_view(n_u, n_u + n_p));
    }

    // take care of constraints on preconditioner and system
    {
      const FEValuesExtractors::Vector velocity(0);
      const FEValuesExtractors::Scalar pressure(dim);

      // system constraints
      VelocityBoundaryValues<dim> velocity_bc_pres;
      constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      // take care of constraints for preconditioner
      preconditioner_constraints.clear();
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              preconditioner_constraints);

//      DoFTools::make_zero_boundary_constraints(dof_handler,
//                                               1,
//                                               preconditioner_constraints,
//                                               fe.component_mask(pressure));
      preconditioner_constraints.close();
    }

    setup_system_matrix(partitioning, relevant_partitioning);
    setup_approx_schur_complement(partitioning, relevant_partitioning);

    solution.reinit(partitioning, MPI_COMM_WORLD);
    solution_primary_problem.reinit(partitioning, MPI_COMM_WORLD);
    temp_vec.reinit(partitioning, MPI_COMM_WORLD);
    system_rhs.reinit(partitioning, MPI_COMM_WORLD);
    solution_distributed.reinit(partitioning,
                                relevant_partitioning,
                                MPI_COMM_WORLD);
    solution_primary_distributed.reinit(partitioning,
                                        relevant_partitioning,
                                        MPI_COMM_WORLD);
  }

  // ------------- solver ----------------------
  template <int dim>
  void
  Darcy<dim>::solve()
  {
    TimerOutput::Scope timer_section(computing_timer, "   Solve system");
    const auto        &M    = system_matrix.block(0, 0);
    const auto        &ap_S = precondition_matrix.block(1, 1);

    // --------------------------- approx inverse M
    // ------------------------------------- Preconditioner M as incomplete
    // Cholesky decomposition
    TrilinosWrappers::PreconditionIC ap_M_inv;
    ap_M_inv.initialize(M);

    // ------------------ approx inverse Schur as incomplete Cholesky
    // ---------------------
    TrilinosWrappers::PreconditionIC ap_S_inv;
    ap_S_inv.initialize(ap_S);
    const Preconditioner::InverseMatrix<TrilinosWrappers::SparseMatrix,
                                        decltype(ap_S_inv)>
      op_S_inv(ap_S, ap_S_inv);

    // -------------- construct the final preconditioner operator
    // -----------------------
    const Preconditioner::BlockSchurPreconditioner<decltype(op_S_inv),
                                                   decltype(ap_M_inv)>
      block_preconditioner(system_matrix, op_S_inv, ap_M_inv);
    pcout << "Block preconditioner for the system matrix created." << std::endl;

    // ------------------ construct the final inverse operator for the system
    // -----------
    SolverControl solver_control_system(system_matrix.m(),
                                        1.e-16 * system_rhs.l2_norm());
    solver_control_system.enable_history_data();
    solver_control_system.log_history(true);
    solver_control_system.log_result(true);
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver_system(
      solver_control_system,
      SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(100));

    // ----------- distribute the constraints to the solution vector
    // --------------------
    for (unsigned int i = 0; i < solution.size(); ++i)
      if (constraints.is_constrained(i))
        solution(i) = 0;

    // ------------------ solve the system
    // ------------------------------------------------
    TrilinosWrappers::MPI::BlockVector distributed_solution(system_rhs);
    distributed_solution = solution;
    const unsigned int start =
                         (distributed_solution.block(0).size() +
                          distributed_solution.block(1).local_range().first),
                       end =
                         (distributed_solution.block(0).size() +
                          distributed_solution.block(1).local_range().second);

    for (unsigned int i = start; i < end; ++i)
      if (constraints.is_constrained(i))
        distributed_solution(i) = 0;

    pcout << "Starting iterative solver..." << std::endl;
    solver_system.solve(system_matrix,
                        distributed_solution,
                        system_rhs,
                        block_preconditioner);
    constraints.distribute(distributed_solution);
    solution = distributed_solution;

    pcout << solver_control_system.last_step()
          << " GMRES iterations to obtain convergence." << std::endl;
  }

  template <int dim>
  void
  Darcy<dim>::write_data_to_npy(const std::string   &filename,
                                std::vector<double> &data,
                                const unsigned int   rows,
                                const unsigned int   columns) const
  {
    const std::vector<long unsigned> shape{rows, columns};
    const bool                       fortran_order{false};
    npy::SaveArrayAsNumpy(
      filename, fortran_order, shape.size(), shape.data(), data);
  }

} // end of namespace darcy

#endif