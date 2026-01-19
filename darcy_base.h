// darcy_base.h
// Base class for the parallel Darcy flow solver.
// Solves mixed formulation: K^{-1} u + grad(p) = 0, div(u) = f
// with random permeability field K = exp(x) for uncertainty quantification.

#ifndef DARCY_BASE_H
#define DARCY_BASE_H

// === deal.II base includes ===
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

// === Distributed triangulation and grid ===
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

// === Finite elements and DOF handling ===
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_q.h>

// === Linear algebra (serial) ===
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

// === Trilinos parallel linear algebra ===
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// === Numerics (output, assembly helpers) ===
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// === Standard library ===
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <string>

// === External: NumPy file I/O ===
#include "npy.hpp"

// === Local helpers ===
#include "parameters.h"
#include "preconditioner.h"
#include "random_permeability.h"

using namespace dealii;

namespace darcy
{
  // ===========================================================================
  // DarcyBase class: MPI-parallel mixed finite element solver base class.
  //
  // Features:
  //   - Mixed FE: Raviart-Thomas-like velocity (Q_{k+1}) + pressure (Q_k)
  //   - Random permeability field K = exp(x) on separate scalar FE space
  //   - Weak boundary conditions via Nitsche's method
  //   - Block Schur complement preconditioner for saddle-point system
  // ===========================================================================
  template <int dim>
  class DarcyBase
  {
  public:
    // Constructor: degree_p is the polynomial degree for pressure (velocity =
    // degree_p + 1)
    explicit DarcyBase(const unsigned int degree_p);

    // Main entry point for simulation (pure virtual - implemented by derived classes)
    virtual void
    run(const Parameters &params) = 0;

    // Virtual destructor for proper cleanup
    virtual ~DarcyBase() = default;

  protected:
    // -------------------------------------------------------------------------
    // Setup methods
    // -------------------------------------------------------------------------
    void
    setup_grid_and_dofs(); // Create mesh, distribute DOFs, init matrices
    void
    setup_system_matrix(const std::vector<IndexSet> &partitioning,
                        const std::vector<IndexSet> &relevant_partitioning);
    void
    setup_approx_schur_complement(
      const std::vector<IndexSet> &partitioning,
      const std::vector<IndexSet> &relevant_partitioning);

    // -------------------------------------------------------------------------
    // Assembly methods
    // -------------------------------------------------------------------------
    void
    assemble_system(); // Assemble Darcy saddle-point system
    void
    assemble_approx_schur_complement(); // Assemble preconditioner matrix

    // -------------------------------------------------------------------------
    // Solver
    // -------------------------------------------------------------------------
    void
    solve(const bool solve_adjoint = false); // GMRES with block preconditioner

    // -------------------------------------------------------------------------
    // Input methods
    // -------------------------------------------------------------------------
    void
    read_input_npy(
      const std::string &input_path); // Read random field from .npy
    void
    generate_ref_input(); // Generate test random field
    void
    generate_coordinates(); // Setup observation points

    // -------------------------------------------------------------------------
    // Output methods
    // -------------------------------------------------------------------------
    void
    write_data_to_npy(const std::string   &filename,
                      std::vector<double> &data,
                      unsigned int         rows,
                      unsigned int         columns) const;

    // =========================================================================
    // Member variables
    // =========================================================================

    // --- Polynomial degrees ---
    const unsigned int degree_p; // Pressure degree
    const unsigned int degree_u; // Velocity degree (= degree_p + 1)

    // --- Mesh ---
    parallel::distributed::Triangulation<dim>
      triangulation; // Main mesh (MPI-parallel)
    Triangulation<dim>
      triangulation_obs; // Observation mesh (serial, for point evaluation)

    // --- Finite elements and DOF handlers ---
    FESystem<dim>   fe;          // Mixed FE: [Q_{degree_u}]^dim x Q_{degree_p}
    DoFHandler<dim> dof_handler; // DOF handler for velocity-pressure

    FESystem<dim>   rf_fe_system;   // Scalar FE for random field (Q1)
    DoFHandler<dim> rf_dof_handler; // DOF handler for random field

    // --- Constraints ---
    AffineConstraints<double> constraints;                // For system matrix
    AffineConstraints<double> preconditioner_constraints; // For preconditioner

    // --- System matrices ---
    TrilinosWrappers::BlockSparseMatrix system_matrix; // Saddle-point matrix
    TrilinosWrappers::BlockSparseMatrix
      precondition_matrix; // Schur preconditioner

    // --- Solution vectors ---
    TrilinosWrappers::MPI::BlockVector solution; // Current solution [u, p]
    TrilinosWrappers::MPI::BlockVector
      solution_distributed;                        // With ghost values
    TrilinosWrappers::MPI::BlockVector system_rhs; // Right-hand side

    // --- Random field vectors ---
    TrilinosWrappers::MPI::Vector x_vec; // Log-permeability (owned DOFs)
    TrilinosWrappers::MPI::Vector
      x_vec_distributed;                    // Log-permeability (with ghosts)
    TrilinosWrappers::MPI::Vector  mean_rf; // Prior mean (typically zero)
    TrilinosWrappers::SparseMatrix rf_laplace_matrix; // GMRF precision matrix L

    // --- Random field index sets ---
    IndexSet rf_locally_owned;    // DOFs owned by this MPI rank
    IndexSet rf_locally_relevant; // DOFs needed (owned + ghost)

    // --- Observation data ---
    std::vector<Point<dim>> spatial_coordinates; // Observation locations

    // --- I/O and timing ---
    ConditionalOStream pcout;           // Only rank 0 prints
    TimerOutput        computing_timer; // Performance profiling

    // --- Simulation parameters ---
    Parameters params; // Configuration parameters
  };

} // namespace darcy

// ===========================================================================
// Template implementations
// ===========================================================================
namespace darcy
{
  template <int dim>
  void
  DarcyBase<dim>::generate_ref_input()
  {
    const RandomMedium::RefScalar<dim> ref_scalar;
    VectorTools::interpolate(this->rf_dof_handler, ref_scalar, this->x_vec);
  }

  template <int dim>
  void
  DarcyBase<dim>::generate_coordinates()
  {
    this->spatial_coordinates.resize(this->triangulation_obs.n_vertices());
    for (const auto &cell : this->triangulation_obs.active_cell_iterators())
      {
        for (unsigned int v = 0; v < cell->n_vertices(); ++v)
          this->spatial_coordinates[cell->vertex_index(v)] = cell->vertex(v);
      }

    this->pcout << "Number of observation points: "
                << this->spatial_coordinates.size() << std::endl;
  }

  template <int dim>
  DarcyBase<dim>::DarcyBase(const unsigned int degree_p)
    : degree_p(degree_p)
    , degree_u(degree_p + 1)
    , triangulation(MPI_COMM_WORLD,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , triangulation_obs()
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

  template <int dim>
  void
  DarcyBase<dim>::read_input_npy(const std::string &filename)
  {
    TimerOutput::Scope timer_section(this->computing_timer, "   Read Inputs");

    std::vector<unsigned long> shape{};
    bool                       fortran_order{};

    std::vector<double> x_std_vec;
    npy::LoadArrayFromNumpy(filename, shape, fortran_order, x_std_vec);

    const unsigned int n_dofs_rf = this->rf_dof_handler.n_dofs();
    this->pcout << "Read in random field from file: " << filename << std::endl;
    this->pcout << "Number of random field dofs: " << n_dofs_rf << std::endl;
    this->pcout << "Number of input field dofs: " << x_std_vec.size()
                << std::endl;

    TrilinosWrappers::MPI::Vector x_owned(this->rf_locally_owned, MPI_COMM_WORLD);
    for (const auto i : this->rf_locally_owned)
      x_owned[i] = x_std_vec[i];
    x_owned.compress(VectorOperation::insert);
    this->x_vec = x_owned;

    this->pcout << "Random field successfully read in." << std::endl;
  }

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
                                     const unsigned int) const
  {
    return 0.0;
  }

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

    const MappingQ<dim> mapping(2);

    FEValues<dim> fe_values(mapping,
                            this->fe,
                            quadrature_formula,
                            update_JxW_values | update_values |
                              update_quadrature_points | update_gradients);
    FEFaceValues<dim> fe_face_values(mapping,
                                     this->fe,
                                     face_quadrature_formula,
                                     update_values | update_gradients |
                                       update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    FEValues<dim> fe_rf_values(mapping,
                               this->rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    const unsigned int dofs_per_cell   = this->fe.n_dofs_per_cell();
    const unsigned int n_q_points      = fe_values.n_quadrature_points;
    const unsigned int n_face_q_points = fe_face_values.n_quadrature_points;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    this->x_vec_distributed = this->x_vec;

    for (const auto &cell_tria : this->triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(this->dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(this->rf_dof_handler);

        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);
            fe_values.reinit(cell);
            fe_rf_values.reinit(rf_cell);

            cell->get_dof_indices(local_dof_indices);

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);

            local_matrix = 0;

            std::vector<double> rf_values(n_q_points);
            Tensor<2, dim>      K_mat;
            fe_rf_values.get_function_values(this->x_vec_distributed, rf_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                RandomMedium::get_k_mat(rf_values[q], K_mat);

                const auto                  JxW_q = fe_values.JxW(q);
                std::vector<Tensor<1, dim>> grad_phi_p(dofs_per_cell);
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  grad_phi_p[k] = fe_values[pressure].gradient(k, q);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  for (unsigned int j = 0; j <= i; ++j)
                    local_matrix(i, j) +=
                      (K_mat * grad_phi_p[i] * grad_phi_p[j]) * JxW_q;
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
                local_matrix(i, j) = local_matrix(j, i);

            for (const auto &face : cell->face_iterators())
              {
                if (face->at_boundary() && face->boundary_id() == 1)
                  {
                    fe_face_values.reinit(cell, face);

                    const auto tau = 5. *
                                     Utilities::fixed_power<2>(this->degree_p + 1) /
                                     cell->diameter();

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        const auto normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const double phi_i_p =
                              fe_face_values[pressure].value(i, q);

                            const auto grad_phi_i_p =
                              fe_face_values[pressure].gradient(i, q);

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
                              }
                          }
                      }
                  }
              }

            this->preconditioner_constraints.distribute_local_to_global(
              local_matrix, local_dof_indices, this->precondition_matrix);
          }
      }

    this->precondition_matrix.compress(VectorOperation::add);
    this->pcout << "Preconditioner successfully assembled" << std::endl;
  }

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

    const MappingQ<dim> mapping(2);

    FEValues<dim> fe_values(mapping,
                            this->fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);
    FEValues<dim> fe_rf_values(mapping,
                               this->rf_fe_system,
                               quadrature_formula,
                               update_values | update_quadrature_points);
    FEFaceValues<dim> fe_face_values(mapping,
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

    for (const auto &cell_tria : this->triangulation.active_cell_iterators())
      {
        const auto &cell = cell_tria->as_dof_handler_iterator(this->dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(this->rf_dof_handler);

        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_rf_values.reinit(rf_cell);
            cell->get_dof_indices(local_dof_indices);

            const FEValuesExtractors::Vector velocities(0);
            const FEValuesExtractors::Scalar pressure(dim);

            local_matrix = 0;
            local_rhs    = 0;

            std::vector<double> boundary_values_pressure(n_face_q_points);
            const PressureBoundaryValues<dim> pressure_boundary_values;

            std::vector<double> rf_values(n_q_points);
            Tensor<2, dim>      K_mat;
            fe_rf_values.get_function_values(this->x_vec_distributed, rf_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                RandomMedium::get_k_mat(rf_values[q], K_mat);
                const Tensor<2, dim> k_inverse = invert(K_mat);

                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k]      = fe_values[velocities].value(k, q);
                    div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                    phi_p[k]      = fe_values[pressure].value(k, q);
                    grad_phi_p[k] = fe_values[pressure].gradient(k, q);
                  }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        local_matrix(i, j) +=
                          (phi_u[i] * k_inverse * phi_u[j] -
                           phi_p[j] * div_phi_u[i] + grad_phi_p[i] * phi_u[j]) *
                          fe_values.JxW(q);
                      }

                    local_rhs(i) += (-phi_p[i] * 1.0) * fe_values.JxW(q);
                  }
              }

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
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const Tensor<1, dim> phi_i_u =
                              fe_face_values[velocities].value(i, q);

                            const double phi_i_p =
                              fe_face_values[pressure].value(i, q);

                            local_rhs(i) += -(phi_i_u *
                                              fe_face_values.normal_vector(q) *
                                              boundary_values_pressure[q] *
                                              fe_face_values.JxW(q));

                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const Tensor<1, dim> phi_j_u =
                                  fe_face_values[velocities].value(j, q);

                                local_matrix(i, j) -=
                                  (phi_i_p * (fe_face_values.normal_vector(q) *
                                              phi_j_u)) *
                                  fe_face_values.JxW(q);
                              }
                          }
                      }
                  }

                if (face->at_boundary() && face->boundary_id() == 0)
                  {
                    fe_face_values.reinit(cell, face);

                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                      {
                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          {
                            const Tensor<1, dim> phi_i_u =
                              fe_face_values[velocities].value(i, q);

                            for (unsigned int j = 0; j < dofs_per_cell; ++j)
                              {
                                const double phi_j_p =
                                  fe_face_values[pressure].value(j, q);

                                local_matrix(i, j) +=
                                  (phi_j_p *
                                   (phi_i_u * fe_face_values.normal_vector(q)) *
                                   fe_face_values.JxW(q));
                              }
                          }
                      }
                  }
              }

            cell->get_dof_indices(local_dof_indices);
            this->constraints.distribute_local_to_global(local_matrix,
                                                         local_rhs,
                                                         local_dof_indices,
                                                         this->system_matrix,
                                                         this->system_rhs);
          }
      }

    this->system_matrix.compress(VectorOperation::add);
    this->system_rhs.compress(VectorOperation::add);

    this->pcout << "System successfully assembled" << std::endl;
  }

  template <int dim>
  void
  DarcyBase<dim>::setup_system_matrix(
    const std::vector<IndexSet> &partitioning,
    const std::vector<IndexSet> &relevant_partitioning)
  {
    this->system_matrix.clear();

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
  }

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

  template <int dim>
  void
  DarcyBase<dim>::setup_grid_and_dofs()
  {
    TimerOutput::Scope timing_section(this->computing_timer, "Setup dof systems");
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;

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

    GridGenerator::eccentric_hyper_shell(this->triangulation,
                                         inner_center,
                                         outer_center,
                                         inner_radius,
                                         outer_radius,
                                         n_cells);

    GridGenerator::eccentric_hyper_shell(this->triangulation_obs,
                                         inner_center,
                                         outer_center,
                                         inner_radius + 0.05,
                                         outer_radius - 0.05,
                                         n_cells);
    this->triangulation_obs.refine_global(3);

    this->triangulation.refine_global(4);
    this->dof_handler.distribute_dofs(this->fe);
    DoFRenumbering::Cuthill_McKee(this->dof_handler);
    DoFRenumbering::component_wise(this->dof_handler, block_component);

    this->rf_dof_handler.distribute_dofs(this->rf_fe_system);

    this->rf_locally_owned = this->rf_dof_handler.locally_owned_dofs();
    this->rf_locally_relevant =
      DoFTools::extract_locally_relevant_dofs(this->rf_dof_handler);

    this->x_vec.reinit(this->rf_locally_owned, MPI_COMM_WORLD);
    this->x_vec_distributed.reinit(this->rf_locally_owned,
                                   this->rf_locally_relevant,
                                   MPI_COMM_WORLD);

    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler, block_component);

    const types::global_dof_index n_u = dofs_per_block[0],
                                  n_p = dofs_per_block[1];

    std::locale s = this->pcout.get_stream().getloc();
    this->pcout.get_stream().imbue(std::locale(""));
    this->pcout << "Number of active cells: " << this->triangulation.n_active_cells()
                << std::endl
                << "Total number of cells: " << this->triangulation.n_cells()
                << std::endl
                << "Number of degrees of freedom: " << this->dof_handler.n_dofs()
                << " (" << n_u << '+' << n_p << ')' << std::endl
                << "Number of random field dofs: " << this->rf_dof_handler.n_dofs()
                << std::endl;
    this->pcout.get_stream().imbue(s);

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

    {
      const FEValuesExtractors::Vector velocity(0);
      const FEValuesExtractors::Scalar pressure(dim);

      this->constraints.clear();
      this->constraints.reinit(relevant_set);
      DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
      this->constraints.close();

      this->preconditioner_constraints.clear();
      this->preconditioner_constraints.reinit(relevant_set);
      DoFTools::make_hanging_node_constraints(this->dof_handler,
                                              this->preconditioner_constraints);

      this->preconditioner_constraints.close();
    }

    setup_system_matrix(partitioning, relevant_partitioning);
    setup_approx_schur_complement(partitioning, relevant_partitioning);

    this->solution.reinit(partitioning, MPI_COMM_WORLD);
    this->system_rhs.reinit(partitioning, MPI_COMM_WORLD);
    this->solution_distributed.reinit(partitioning,
                                      relevant_partitioning,
                                      MPI_COMM_WORLD);
  }

  template <typename MatrixType>
  class TransposeOperator : public Subscriptor
  {
  public:
    TransposeOperator(const MatrixType &matrix)
      : matrix(matrix)
    {}

    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      matrix.Tvmult(dst, src);
    }

    template <typename VectorType>
    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      matrix.vmult(dst, src);
    }

    types::global_dof_index
    m() const
    {
      return matrix.n();
    }

    types::global_dof_index
    n() const
    {
      return matrix.m();
    }

  private:
    const MatrixType &matrix;
  };

  template <int dim>
  void
  DarcyBase<dim>::solve(const bool adjoint_solve)
  {
    TimerOutput::Scope timer_section(this->computing_timer, "   Solve system");
    const auto        &M    = this->system_matrix.block(0, 0);
    const auto        &ap_S = this->precondition_matrix.block(1, 1);

    TrilinosWrappers::PreconditionIC ap_M_inv;
    ap_M_inv.initialize(M);

    TrilinosWrappers::PreconditionIC ap_S_inv;
    ap_S_inv.initialize(ap_S);
    const Preconditioner::InverseMatrix<TrilinosWrappers::SparseMatrix,
                                        decltype(ap_S_inv)>
      op_S_inv(ap_S, ap_S_inv);

    const Preconditioner::BlockSchurPreconditioner<decltype(op_S_inv),
                                                   decltype(ap_M_inv)>
      block_preconditioner(this->system_matrix, op_S_inv, ap_M_inv,
                           this->computing_timer);
    this->pcout << "Block preconditioner for the system matrix created."
                << std::endl;

    const double rhs_norm = this->system_rhs.l2_norm();
    const double abs_tol  = 1.e-14;
    const double rel_reduction = 1.e-12;

    this->pcout << "Solver: abs_tol=" << abs_tol
                << ", rel_reduction=" << rel_reduction
                << ", RHS norm=" << rhs_norm << std::endl;

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

    this->solution = 0;

    TrilinosWrappers::MPI::BlockVector distributed_solution(this->system_rhs);
    distributed_solution = 0;

    this->pcout << "Starting iterative solver..." << std::endl;
    {
      TimerOutput::Scope timer_section(this->computing_timer, "   Solve gmres");
      if (adjoint_solve)
        {
          TransposeOperator<decltype(this->system_matrix)> system_transposed(
            this->system_matrix);

          TransposeOperator<decltype(block_preconditioner)>
            preconditioner_transposed(block_preconditioner);

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
} // namespace darcy

#endif // DARCY_BASE_H
