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
    VectorTools::interpolate(rf_dof_handler, ref_scalar, x_vec);
  }

  // generate coordinates at which observations are present
  template <int dim>
  void
  Darcy<dim>::generate_coordinates()
  {
    // triangulation_obs is a serial Triangulation, so all ranks have the
    // complete mesh with all vertices. Simply collect all unique vertices.
    spatial_coordinates.resize(triangulation_obs.n_vertices());
    for (const auto &cell : triangulation_obs.active_cell_iterators())
      {
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          {
            spatial_coordinates[cell->vertex_index(v)] = cell->vertex(v);
          }
      }


    pcout << "Number of observation points: " << spatial_coordinates.size()
          << std::endl;
  }

  // ------ class constructor ----------------
  template <int dim>
  Darcy<dim>::Darcy(const unsigned int degree_p)
    : degree_p(degree_p)
    , degree_u(degree_p + 1)
    , triangulation(MPI_COMM_WORLD,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , triangulation_obs() // serial triangulation for observation points
    , fe(FE_Q<dim>(degree_u), dim, FE_Q<dim>(degree_p), 1)
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

    // Read the full vector on all ranks (file I/O is typically fast)
    std::vector<double> x_std_vec;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, x_std_vec);

    const unsigned int n_dofs_rf = rf_dof_handler.n_dofs();
    pcout << "Read in random field from file: " << filename << std::endl;
    pcout << "Number of random field dofs: " << n_dofs_rf << std::endl;
    pcout << "Number of input field dofs: " << x_std_vec.size() << std::endl;

    // Create owned-only temporary vector, fill it, then assign to x_vec
    // (the assignment to a ghosted vector triggers ghost value communication)
    TrilinosWrappers::MPI::Vector x_owned(rf_locally_owned, MPI_COMM_WORLD);
    for (const auto i : rf_locally_owned)
      x_owned[i] = x_std_vec[i];
    x_owned.compress(VectorOperation::insert);
    x_vec = x_owned; // Assignment to ghosted vector updates ghost values

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
  PressureBoundaryValues<dim>::value(const Point<dim> &,
                                     const unsigned int /*component*/) const
  {
    return 0.0;
  }


  // ------------- assemble approx Schur complement ---------
  template <int dim>
  void
  Darcy<dim>::assemble_approx_schur_complement()
  {
    TimerOutput::Scope timer_section(computing_timer,
                                     "   Assemble approx. Schur compl.");
    pcout << "Assemble approx. Schur complement..." << std::endl;
    precondition_matrix = 0;
    const QGauss<dim>     quadrature_formula(degree_p + 1);
    const QGauss<dim - 1> face_quadrature_formula(degree_p + 1);

    // Use higher-order mapping for curved geometry (eccentric_hyper_shell)
    const MappingQ<dim> mapping(1);

    // start the cell loop
    FEValues<dim>      fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_JxW_values | update_values |
                              update_quadrature_points | update_gradients);
    FEFaceValues<dim>  fe_face_values(mapping,
                                     fe,
                                     face_quadrature_formula,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    FEValues<dim>      fe_rf_values(mapping,
                               rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    x_vec_distributed = x_vec; // make sure ghost values are updated

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
            fe_rf_values.get_function_values(x_vec_distributed, rf_values);

            // quadrature loop
            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                // evaluate random field at quadrature point
                RandomMedium::get_k_mat(rf_values[q], K_mat);

                // evaluate fe values on all dofs first
                const auto                  JxW_q = fe_values.JxW(q);
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
              } // end quadrature loop

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
                    const auto tau = 5. *
                                     Utilities::fixed_power<2>(degree_p + 1) /
                                     cell->diameter();

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

                                local_matrix(i, j) +=
                                  (-grad_phi_i_p * normal * phi_j_p -
                                   (phi_i_p *
                                    (grad_phi_j_p * normal - tau * phi_j_p))) *
                                  fe_face_values.JxW(q);
                              } // end inner dof loop j
                          } // end face dof loops i
                      } // end quadrature loop for faces
                  } // end if statement
              } // end face loop

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

    const QGauss<dim>     quadrature_formula(degree_u + 1);
    const QGauss<dim - 1> face_quadrature_formula(degree_u + 1);

    // Use higher-order mapping for curved geometry (eccentric_hyper_shell)
    const MappingQ<dim> mapping(1);

    FEValues<dim>      fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEValues<dim>      fe_rf_values(mapping,
                               rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    FEFaceValues<dim>  fe_face_values(mapping,
                                     fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;
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
            fe_rf_values.get_function_values(x_vec_distributed, rf_values);

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
                          } // end face dof loops i
                      } // end quadrature loop for faces
                  } // end if statement

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
                          } // end face dof loops i
                      } // end quadrature loop for faces
                  } // end if statement
              } // end face loop

            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);

          } // end if locally owned
      } // end cell loop

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
        if (((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::none;
        else
          {
            if (c < dim && d < dim)
              {
                if (c == d)
                  coupling[c][d] = DoFTools::always;
                else
                  coupling[c][d] = DoFTools::none;
              }
            else
              coupling[c][d] = DoFTools::always;
          }

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

  // ---------- setup approx schur complement ------------------------
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
    for (unsigned int c = dim; c < dim + 1; ++c)
      for (unsigned int d = dim; d < dim + 1; ++d)
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

    double       inner_radius = 0.3; // inner radius
    double       outer_radius = 1.0; // outer radius
    unsigned int n_cells      = 12;  // n_cells

    // GridGenerator::eccentric_hyper_shell(triangulation,
    //                                      inner_center,
    //                                      outer_center,
    //                                      inner_radius,
    //                                      outer_radius,
    //                                      n_cells);

    GridGenerator::hyper_cube(triangulation, 0, 1, true);

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        // In a parallel triangulation, we only modify cells we 'own' or 'ghost'
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
             ++f)
          {
            if (cell->face(f)->at_boundary() and
                cell->face(f)->boundary_id() != 0)
              {
                cell->face(f)->set_boundary_id(1);
              }
          }
      }
    GridGenerator::hyper_cube(triangulation_obs, 0.01, 0.99, true);
    for (const auto &cell : triangulation_obs.active_cell_iterators())
      {
        // In a parallel triangulation, we only modify cells we 'own' or 'ghost'
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell;
             ++f)
          {
            if (cell->face(f)->at_boundary() and
                cell->face(f)->boundary_id() != 0)
              {
                cell->face(f)->set_boundary_id(1);
              }
          }
      }
    // // introduce coarse tria/grid for artificial observations only
    // GridGenerator::eccentric_hyper_shell(triangulation_obs,
    //                                      inner_center,
    //                                      outer_center,
    //                                      inner_radius + 0.05,
    //                                      outer_radius - 0.05,
    //                                      n_cells); // TODO: hack to have
    //                                      different mesh
    triangulation_obs.refine_global(5);

    // now the actual grid for the forward problem
    triangulation.refine_global(5); // 4
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(
      dof_handler); // Cuthill_McKee, component_wise to be more efficient
    DoFRenumbering::component_wise(dof_handler, block_component);

    // generate grid and distribute dofs for random field
    rf_dof_handler.distribute_dofs(rf_fe_system);
    // DoFRenumbering::Cuthill_McKee(
    //   rf_dof_handler); // Cuthill_McKee, component_wise to be more efficient

    // NOTE: Do NOT apply DoFRenumbering here (e.g., Cuthill_McKee).
    // The input .npy files contain values in a specific global DOF order
    // that was determined when the file was created. Any renumbering would
    // change the DOF-to-physical-location mapping, causing x_vec[i] to
    // correspond to a different spatial location than intended.
    // If Cuthill-McKee is desired for performance, regenerate all input
    // files with the new numbering.

    // Setup index sets for the random field (needed for parallel x_vec)
    rf_locally_owned = rf_dof_handler.locally_owned_dofs();
    rf_locally_relevant =
      DoFTools::extract_locally_relevant_dofs(rf_dof_handler);

    // Initialize x_vec as a distributed vector with ghost values
    x_vec.reinit(rf_locally_owned, MPI_COMM_WORLD);
    x_vec_distributed.reinit(rf_locally_owned,
                             rf_locally_relevant,
                             MPI_COMM_WORLD);

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

      // system constraints - must reinit with locally relevant DOFs for
      // parallel
      constraints.clear();
      constraints.reinit(relevant_set);
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      // take care of constraints for preconditioner
      preconditioner_constraints.clear();
      preconditioner_constraints.reinit(relevant_set);
      DoFTools::make_hanging_node_constraints(dof_handler,
                                              preconditioner_constraints);

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

  template <typename MatrixType>
  class TransposeOperator : public Subscriptor
  {
  public:
    // Constructor stores a reference to the original matrix
    TransposeOperator(const MatrixType &matrix)
      : matrix(matrix)
    {}

    // The Solver calls vmult, but we execute the Transpose (Tvmult)
    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      matrix.Tvmult(dst, src);
    }

    // Optional: If the solver requires the transpose of this operator
    // (which is the original matrix), we map it back to vmult.
    template <typename VectorType>
    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      matrix.vmult(dst, src);
    }

    // Return dimensions flipped:
    // The number of rows of A^T is the number of columns of A.
    types::global_dof_index
    m() const
    {
      return matrix.n();
    }

    // The number of columns of A^T is the number of rows of A.
    types::global_dof_index
    n() const
    {
      return matrix.m();
    }

  private:
    const MatrixType &matrix;
  };

  // ------------- solver ----------------------
  template <int dim>
  void
  Darcy<dim>::solve(const bool adjoint_solve)
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
      block_preconditioner(system_matrix, op_S_inv, ap_M_inv, computing_timer);
    pcout << "Block preconditioner for the system matrix created." << std::endl;

    // ------------------ construct the final inverse operator for the system
    // -----------
    // Use ReductionControl for both absolute and relative tolerance:
    // - Absolute tolerance: 1e-14 (floor for very small RHS)
    // - Relative tolerance (reduction): 1e-10 (reduce residual by 10 orders of
    // magnitude)
    const double rhs_norm = system_rhs.l2_norm();
    const double abs_tol  = 1.e-14;
    const double rel_reduction =
      1.e-12; // This is the reduction factor, not absolute

    pcout << "Solver: abs_tol=" << abs_tol
          << ", rel_reduction=" << rel_reduction << ", RHS norm=" << rhs_norm
          << std::endl;

    // ReductionControl solver_control_system(system_matrix.m(),
    //                                        abs_tol,
    //                                        rel_reduction);
    dealii::SolverControl solver_control_system(system_matrix.m(),
                                                1.0e-10 * system_rhs.l2_norm(),
                                                true,
                                                1.0e-10);
    solver_control_system.enable_history_data();
    solver_control_system.log_history(true);
    solver_control_system.log_result(true);
    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver_system(
      solver_control_system,
      SolverGMRES<TrilinosWrappers::MPI::BlockVector>::AdditionalData(200));

    // ----------- initialize solution vectors --------------------
    // Zero the solution vector for a deterministic initial guess.
    // Note: We use weak BCs (Nitsche) for both pressure and velocity,
    // so constraints only contain hanging nodes. With global refinement,
    // there are no hanging nodes, so constraints is empty.
    solution = 0;

    // ------------------ solve the system
    // ------------------------------------------------
    TrilinosWrappers::MPI::BlockVector distributed_solution(system_rhs);
    distributed_solution = 0; // Explicit zero initial guess for reproducibility

    pcout << "Starting iterative solver..." << std::endl;
    {
      TimerOutput::Scope timer_section(computing_timer, "   Solve gmres");
      if (adjoint_solve)
        {
          // Solve the adjoint
          // 1. Wrap the System Matrix
          TransposeOperator<decltype(system_matrix)> system_transposed(
            system_matrix);

          // 2. Wrap the Preconditioner
          TransposeOperator<decltype(block_preconditioner)>
            preconditioner_transposed(block_preconditioner);

          // 3. Solve
          solver_system.solve(system_transposed,
                              distributed_solution,
                              system_rhs,
                              preconditioner_transposed);
        }
      else
        {
          solver_system.solve(system_matrix,
                              distributed_solution,
                              system_rhs,
                              block_preconditioner);
        }
    }
    constraints.distribute(distributed_solution);
    solution = distributed_solution;

    pcout << solver_control_system.final_reduction()
          << " GMRES reduction to obtain convergence." << std::endl;

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
