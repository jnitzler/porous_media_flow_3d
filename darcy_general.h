// darcy_general.h
#ifndef DARCY_GENERAL_H
#define DARCY_GENERAL_H

#include "darcy.h"
#include "preconditioner.h"
#include "random_permeability.h"

namespace darcy
{
  // Generate reference permeability field from analytical function (for
  // testing).
  template <int dim>
  void
  DarcyBase<dim>::generate_ref_input()
  {
    const RandomMedium::RefScalar<dim> ref_scalar;
    VectorTools::interpolate(this->rf_dof_handler, ref_scalar, this->x_vec);
  }

  // Populate spatial_coordinates with observation point locations.
  // Uses vertices from triangulation_obs (serial mesh for observations).
  template <int dim>
  void
  DarcyBase<dim>::generate_coordinates()
  {
    // triangulation_obs is a serial Triangulation, so all ranks have the
    // complete mesh with all vertices. Simply collect all unique vertices.
    this->spatial_coordinates.resize(this->triangulation_obs.n_vertices());
    for (const auto &cell : this->triangulation_obs.active_cell_iterators())
      {
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          {
            this->spatial_coordinates[cell->vertex_index(v)] = cell->vertex(v);
          }
      }


    this->pcout << "Number of observation points: " << this->spatial_coordinates.size()
          << std::endl;
  }

  // Constructor: initialize FE systems, triangulation, and MPI communicators.
  template <int dim>
  DarcyBase<dim>::DarcyBase(const unsigned int degree_p)
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

  // Derived class constructors
  template <int dim>
  Darcy<dim>::Darcy(const unsigned int degree_p)
    : DarcyBase<dim>(degree_p)
  {}

  template <int dim>
  DarcyAdjoint<dim>::DarcyAdjoint(const unsigned int degree_p)
    : DarcyBase<dim>(degree_p)
  {}

  // Read log-permeability field from .npy file into x_vec.
  // File must contain rf_dof_handler.n_dofs() values.
  template <int dim>
  void
  DarcyBase<dim>::read_input_npy(const std::string &filename)
  {
    TimerOutput::Scope timer_section(this->computing_timer, "   Read Inputs");

    std::vector<unsigned long> shape{};
    bool                       fortran_order{};

    // Read the full vector on all ranks (file I/O is typically fast)
    std::vector<double> x_std_vec;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, x_std_vec);

    const unsigned int n_dofs_rf = this->rf_dof_handler.n_dofs();
    this->pcout << "Read in random field from file: " << filename << std::endl;
    this->pcout << "Number of random field dofs: " << n_dofs_rf << std::endl;
    this->pcout << "Number of input field dofs: " << x_std_vec.size() << std::endl;

    // Create owned-only temporary vector, fill it, then assign to x_vec
    // (the assignment to a ghosted vector triggers ghost value communication)
    TrilinosWrappers::MPI::Vector x_owned(this->rf_locally_owned, MPI_COMM_WORLD);
    for (const auto i : this->rf_locally_owned)
      x_owned[i] = x_std_vec[i];
    x_owned.compress(VectorOperation::insert);
    this->x_vec = x_owned; // Assignment to ghosted vector updates ghost values

    this->pcout << "Random field successfully read in." << std::endl;
  }


  // Pressure boundary condition function (returns 0 on outer boundary).
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


  // Assemble pressure-Laplacian matrix for Schur complement preconditioner.
  // Uses Nitsche's method for boundary conditions on pressure.
  template <int dim>
  void
  DarcyBase<dim>::assemble_approx_schur_complement()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble approx. Schur compl.");
    this->pcout << "Assemble approx. Schur complement..." << std::endl;
    this->precondition_matrix = 0;
    const QGauss<dim>     quadrature_formula(this->degree_p + 1);
    const QGauss<dim - 1> face_quadrature_formula(this->degree_p + 1);

    // Use higher-order mapping for curved geometry (eccentric_hyper_shell)
    const MappingQ<dim> mapping(2);

    // start the cell loop
    FEValues<dim>      fe_values(mapping,
                            this->fe,
                            quadrature_formula,
                            update_JxW_values | update_values |
                              update_quadrature_points | update_gradients);
    FEFaceValues<dim>  fe_face_values(mapping,
                                     this->fe,
                                     face_quadrature_formula,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    FEValues<dim>      fe_rf_values(mapping,
                               this->rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    const unsigned int dofs_per_cell   = this->fe.n_dofs_per_cell();
    const unsigned int n_q_points      = fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    this->x_vec_distributed = this->x_vec; // make sure ghost values are updated

    for (const auto &cell_tria : this->triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(this->dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(this->rf_dof_handler);

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
            fe_rf_values.get_function_values(this->x_vec_distributed, rf_values);

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
                                     Utilities::fixed_power<2>(this->degree_p + 1) /
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

            this->preconditioner_constraints.distribute_local_to_global(
              local_matrix, local_dof_indices, this->precondition_matrix);

          } // end if locally owned

      } // end cell loop

    this->precondition_matrix.compress(VectorOperation::add);
    this->pcout << "Preconditioner successfully assembled" << std::endl;
  }

  // Assemble the Darcy system matrix and RHS.
  // Mixed formulation: (K^{-1} u, v) - (p, div v) + (grad q, u) = (f, q).
  // Weak BCs: pressure on outer boundary (id=1), zero velocity on inner (id=0).
  template <int dim>
  void
  DarcyBase<dim>::assemble_system()
  {
    TimerOutput::Scope timer_section(this->computing_timer, "  Assemble system");
    this->pcout << "Assemble system..." << std::endl;

    this->system_matrix = 0;
    this->system_rhs    = 0;

    const QGauss<dim>     quadrature_formula(this->degree_u + 1);
    const QGauss<dim - 1> face_quadrature_formula(this->degree_u + 1);

    // Use higher-order mapping for curved geometry (eccentric_hyper_shell)
    const MappingQ<dim> mapping(2);

    FEValues<dim>      fe_values(mapping,
                            this->fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEValues<dim>      fe_rf_values(mapping,
                               this->rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    FEFaceValues<dim>  fe_face_values(mapping,
                                     this->fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    const unsigned int dofs_per_cell   = this->fe.n_dofs_per_cell();
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
    for (const auto &cell_tria : this->triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(this->dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(this->rf_dof_handler);

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
            fe_rf_values.get_function_values(this->x_vec_distributed, rf_values);

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
            this->constraints.distribute_local_to_global(local_matrix,
                                                   local_rhs,
                                                   local_dof_indices,
                                                   this->system_matrix,
                                                   this->system_rhs);

          } // end if locally owned
      } // end cell loop

    this->system_matrix.compress(VectorOperation::add);
    this->system_rhs.compress(VectorOperation::add);

    this->pcout << "System successfully assembled" << std::endl;
  }

  // Setup sparsity pattern and reinit system_matrix for the saddle-point
  // system.
  template <int dim>
  void
  DarcyBase<dim>::setup_system_matrix(
    const std::vector<IndexSet> &partitioning,
    const std::vector<IndexSet> &relevant_partitioning)
  {
    this->system_matrix.clear();
    this->block_mass_matrix.clear();

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

    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    coupling,
                                    sp,
                                    this->constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    sp.compress();

    this->system_matrix.reinit(sp);
    this->block_mass_matrix.reinit(
      sp); // we just use the same sparsity pattern
           // for the block mass matrix (needed for adjoint) here
           // this should be moved to a separate function in the future
  }

  // Setup sparsity pattern for the Schur complement preconditioner matrix.
  template <int dim>
  void
  DarcyBase<dim>::setup_approx_schur_complement(
    const std::vector<IndexSet> &partitioning,
    const std::vector<IndexSet> &relevant_partitioning)
  {
    this->precondition_matrix.clear();

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

    DoFTools::make_sparsity_pattern(this->dof_handler,
                                    coupling_precond,
                                    sp,
                                    this->constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    sp.compress();

    this->precondition_matrix.reinit(sp);
  }

  // Create eccentric_hyper_shell mesh, distribute DOFs, setup constraints and
  // matrices. Generates both the main mesh and a coarser observation mesh
  // (triangulation_obs).
  template <int dim>
  void
  DarcyBase<dim>::setup_grid_and_dofs()
  {
    TimerOutput::Scope timing_section(this->computing_timer, "Setup dof systems");
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

    GridGenerator::eccentric_hyper_shell(this->triangulation,
                                         inner_center,
                                         outer_center,
                                         inner_radius,
                                         outer_radius,
                                         n_cells);

    // introduce coarse tria/grid for artificial observations only
    GridGenerator::eccentric_hyper_shell(this->triangulation_obs,
                                         inner_center,
                                         outer_center,
                                         inner_radius + 0.05,
                                         outer_radius - 0.05,
                                         n_cells);
    this->triangulation_obs.refine_global(3);

    // now the actual grid for the forward problem
    this->triangulation.refine_global(4);
    this->dof_handler.distribute_dofs(this->fe);
    DoFRenumbering::Cuthill_McKee(
      this->dof_handler); // Cuthill_McKee, component_wise to be more efficient
    DoFRenumbering::component_wise(this->dof_handler, block_component);

    // generate grid and distribute dofs for random field
    this->rf_dof_handler.distribute_dofs(this->rf_fe_system);

    // Setup index sets for the random field (needed for parallel x_vec)
    this->rf_locally_owned = this->rf_dof_handler.locally_owned_dofs();
    this->rf_locally_relevant =
      DoFTools::extract_locally_relevant_dofs(this->rf_dof_handler);

    // Initialize x_vec as a distributed vector with ghost values
    this->x_vec.reinit(this->rf_locally_owned, MPI_COMM_WORLD);
    this->x_vec_distributed.reinit(this->rf_locally_owned,
                             this->rf_locally_relevant,
                             MPI_COMM_WORLD);

    // count dofs per block
    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);

    const types::global_dof_index n_u = dofs_per_block[0],
                                  n_p = dofs_per_block[1];

    std::locale s = this->pcout.get_stream().getloc();
    this->pcout.get_stream().imbue(std::locale(""));
    this->pcout << "Number of active cells: " << this->triangulation.n_active_cells()
          << std::endl
          << "Total number of cells: " << this->triangulation.n_cells() << std::endl
          << "Number of degrees of freedom: " << this->dof_handler.n_dofs() << " ("
          << n_u << '+' << n_p << ')' << std::endl
          << "Number of random field dofs: " << this->rf_dof_handler.n_dofs()
          << std::endl;
    this->pcout.get_stream().imbue(s);

    // create relevant index set
    std::vector<IndexSet> partitioning, relevant_partitioning;
    IndexSet              relevant_set;
    {
      IndexSet index_set = this->dof_handler.locally_owned_dofs();
      partitioning.push_back(index_set.get_view(0, n_u));
      partitioning.push_back(index_set.get_view(n_u, n_u + n_p));

      relevant_set = DoFTools::extract_locally_relevant_dofs(this->dof_handler);
      relevant_partitioning.push_back(relevant_set.get_view(0, n_u));
      relevant_partitioning.push_back(relevant_set.get_view(n_u, n_u + n_p));
    }

    // take care of constraints on preconditioner and system
    {
      const FEValuesExtractors::Vector velocity(0);
      const FEValuesExtractors::Scalar pressure(dim);

      // system constraints - must reinit with locally relevant DOFs for
      // parallel
      this->constraints.clear();
      this->constraints.reinit(relevant_set);
      DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
      this->constraints.close();

      // take care of constraints for preconditioner
      this->preconditioner_constraints.clear();
      this->preconditioner_constraints.reinit(relevant_set);
      DoFTools::make_hanging_node_constraints(this->dof_handler,
                                              this->preconditioner_constraints);

      this->preconditioner_constraints.close();
    }

    setup_system_matrix(partitioning, relevant_partitioning);
    setup_approx_schur_complement(partitioning, relevant_partitioning);

    this->solution.reinit(partitioning, MPI_COMM_WORLD);
    this->temp_vec.reinit(partitioning, MPI_COMM_WORLD);
    this->system_rhs.reinit(partitioning, MPI_COMM_WORLD);
    this->solution_distributed.reinit(partitioning,
                                relevant_partitioning,
                                MPI_COMM_WORLD);
  }

  // Helper class: wraps a matrix to provide transpose operations.
  // Used for solving adjoint system A^T x = b.
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

  // Solve the linear system using GMRES with block Schur preconditioner.
  // If adjoint_solve=true, solves A^T x = b instead of A x = b.
  template <int dim>
  void
  DarcyBase<dim>::solve(const bool adjoint_solve)
  {
    TimerOutput::Scope timer_section(this->computing_timer, "   Solve system");
    const auto        &M    = this->system_matrix.block(0, 0);
    const auto        &ap_S = this->precondition_matrix.block(1, 1);

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
      block_preconditioner(this->system_matrix, op_S_inv, ap_M_inv, this->computing_timer);
    this->pcout << "Block preconditioner for the system matrix created." << std::endl;

    // ------------------ construct the final inverse operator for the system
    // -----------
    // Use ReductionControl for both absolute and relative tolerance:
    // - Absolute tolerance: 1e-14 (floor for very small RHS)
    // - Relative tolerance (reduction): 1e-10 (reduce residual by 10 orders of
    // magnitude)
    const double rhs_norm = this->system_rhs.l2_norm();
    const double abs_tol  = 1.e-14;
    const double rel_reduction =
      1.e-12; // This is the reduction factor, not absolute

    this->pcout << "Solver: abs_tol=" << abs_tol
          << ", rel_reduction=" << rel_reduction << ", RHS norm=" << rhs_norm
          << std::endl;

    // ReductionControl solver_control_system(system_matrix.m(),
    //                                        abs_tol,
    //                                        rel_reduction);
    dealii::SolverControl solver_control_system(this->system_matrix.m(),
                                                1.0e-10 * this->system_rhs.l2_norm(),
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
    this->solution = 0;

    // ------------------ solve the system
    // ------------------------------------------------
    TrilinosWrappers::MPI::BlockVector distributed_solution(this->system_rhs);
    distributed_solution = 0; // Explicit zero initial guess for reproducibility

    this->pcout << "Starting iterative solver..." << std::endl;
    {
      TimerOutput::Scope timer_section(this->computing_timer, "   Solve gmres");
      if (adjoint_solve)
        {
          // Solve the adjoint
          // 1. Wrap the System Matrix
          TransposeOperator<decltype(this->system_matrix)> system_transposed(
            this->system_matrix);

          // 2. Wrap the Preconditioner
          TransposeOperator<decltype(block_preconditioner)>
            preconditioner_transposed(block_preconditioner);

          // 3. Solve
          solver_system.solve(system_transposed,
                              distributed_solution,
                              this->system_rhs,
                              preconditioner_transposed);
        }
      else
        {
          solver_system.solve(this->system_matrix,
                              distributed_solution,
                              this->system_rhs,
                              block_preconditioner);
        }
    }
    this->constraints.distribute(distributed_solution);
    this->solution = distributed_solution;

    this->pcout << solver_control_system.final_reduction()
          << " GMRES reduction to obtain convergence." << std::endl;

    this->pcout << solver_control_system.last_step()
          << " GMRES iterations to obtain convergence." << std::endl;
  }

  // Write data vector to .npy file (NumPy format). Only call from rank 0.
  template <int dim>
  void
  DarcyBase<dim>::write_data_to_npy(const std::string   &filename,
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
