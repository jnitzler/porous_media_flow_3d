// darcy_adjoint.h
// Adjoint Darcy solver class for gradient computation.
// Implements the adjoint simulation for gradient-based optimization.

#ifndef DARCY_ADJOINT_H
#define DARCY_ADJOINT_H

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <filesystem>
#include <map>

#include "darcy_base.h"

namespace darcy
{
  // ===========================================================================
  // DarcyAdjoint class: Adjoint Darcy solver for gradient computation
  // ===========================================================================
  template <int dim>
  class DarcyAdjoint : public DarcyBase<dim>
  {
  public:
    // Constructor
    explicit DarcyAdjoint(const unsigned int degree_p,
                          const unsigned int degree_rf);

    // Main entry point for adjoint simulation
    void
    run(const Parameters &params) override;

  private:
    // -------------------------------------------------------------------------
    // Adjoint-specific input methods
    // -------------------------------------------------------------------------
    void
    read_primary_solution(); // Load forward solution
    void
    read_upstream_gradient_npy(
      const std::string &input_file_path); // Load dL/dy

    // -------------------------------------------------------------------------
    // Adjoint-specific methods
    // -------------------------------------------------------------------------
    void
    setup_point_evaluation_cache(); // Initialize RPE for observation points
    void
    overwrite_adjoint_rhs(); // Build RHS from upstream gradient (mollified)
    void
    final_inner_adjoint_product(); // Compute dL/dx = -lambda^T (dA/dx) u
    void
    create_rf_laplace_operator(); // GMRF prior precision matrix
    void
    add_prior_gradient_to_adjoint(); // Add prior regularization to gradient

    // -------------------------------------------------------------------------
    // Helper functions
    // -------------------------------------------------------------------------
    double
    compute_average_cell_diameter() const;
    bool
    is_cell_near_any_observation(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const double                                          cutoff) const;

    // -------------------------------------------------------------------------
    // Adjoint-specific output methods
    // -------------------------------------------------------------------------
    void
    write_adjoint_solution_pvtu();
    void
    output_gradient_pvtu();
    void
    write_gradient_to_npy();

    // -------------------------------------------------------------------------
    // Simulation driver
    // -------------------------------------------------------------------------
    void
    run_simulation();

    // =========================================================================
    // Adjoint-specific member variables
    // =========================================================================

    // --- Adjoint-specific vectors ---
    TrilinosWrappers::MPI::BlockVector
      solution_primary_problem; // Forward solution
    TrilinosWrappers::MPI::BlockVector
      solution_primary_distributed; // With ghosts
    std::vector<double>
      grad_log_lik_x_distributed;       // Gradient (local contributions)
    std::vector<double> grad_log_lik_x; // Gradient (gathered on all ranks)
    FullMatrix<double>  grad_pde_x;     // Reserved for future use

    // --- Adjoint observation data ---
    std::vector<double> adjoint_data_vec; // Upstream gradient dL/dy (flat)
    std::vector<std::vector<double>>
                        data_vec; // Upstream gradient [point][component]
    std::vector<double> grad_log_lik_x_partial_distributed; // Partial gradient

    // --- Cached point evaluation data (from
    // distributed_compute_point_locations) ---
    std::vector<typename DoFHandler<dim>::active_cell_iterator> cached_cells;
    std::vector<std::vector<Point<dim>>> cached_qpoints; // unit points
    std::vector<std::vector<unsigned int>>
      cached_point_indices; // original indices
  };

  // Constructor implementation
  template <int dim>
  DarcyAdjoint<dim>::DarcyAdjoint(const unsigned int degree_p,
                                  const unsigned int degree_rf)
    : DarcyBase<dim>(degree_p, degree_rf)
  {}

  // ========================================================================
  // Template implementations
  // ========================================================================

  template <int dim>
  void
  DarcyAdjoint<dim>::read_upstream_gradient_npy(
    const std::string &input_file_path)
  {
    TimerOutput::Scope    timing_section(this->computing_timer,
                                      "read upstream gradient npy");
    std::filesystem::path my_path(input_file_path);
    std::filesystem::path base_dir = my_path.parent_path();

    const std::string adjoint_file_path =
      base_dir.string() + "/adjoint_data.npy";
    this->pcout << "Reading adjoint data from: " << adjoint_file_path
                << std::endl;

    std::vector<unsigned long> shape{};
    bool                       fortran_order;

    npy::LoadArrayFromNumpy(adjoint_file_path,
                            shape,
                            fortran_order,
                            adjoint_data_vec);

    unsigned int len_vec      = adjoint_data_vec.size();
    unsigned int num_data     = len_vec / (dim);
    unsigned int spatial_size = this->spatial_coordinates.size();
    data_vec.resize(spatial_size, std::vector<double>(dim));

    AssertThrow(num_data == spatial_size,
                ExcMessage("Mismatch: num_data=" + std::to_string(num_data) +
                           " vs spatial_size=" + std::to_string(spatial_size)));

    for (unsigned int k = 0; k < spatial_size; ++k)
      for (unsigned int i = 0; i < dim; ++i)
        data_vec[k][i] = adjoint_data_vec[i * num_data + k];
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::read_primary_solution()
  {
    TimerOutput::Scope         timing_section(this->computing_timer,
                                      "read primary solution npy");
    std::vector<unsigned long> shape{};
    bool                       fortran_order;

    std::string file_path = this->params.output_directory + "/" +
                            this->params.output_prefix + "solution_full.npy";
    this->pcout << "Reading primary solution from " << file_path << std::endl;

    std::vector<double> tmp_primary_solution;
    tmp_primary_solution.resize(this->dof_handler.n_dofs());

    npy::LoadArrayFromNumpy(file_path,
                            shape,
                            fortran_order,
                            tmp_primary_solution);

    this->pcout << "Primary solution read successfully!" << std::endl;
    this->pcout << "Writing primary solution to distributed solution vector..."
                << std::endl;

    for (unsigned int i = 0; i < this->dof_handler.n_dofs(); ++i)
      if (solution_primary_problem.in_local_range(i))
        solution_primary_problem[i] = tmp_primary_solution[i];

    solution_primary_problem.compress(VectorOperation::insert);
    this->pcout
      << "Primary solution successfully written to distributed solution vector"
      << std::endl;
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::setup_point_evaluation_cache()
  {
    TimerOutput::Scope timing_section(this->computing_timer,
                                      "Setup point evaluation cache");
    this->pcout << "Setting up point evaluation cache..." << std::endl;

    const MappingQ<dim> mapping(2);

    cached_cells.clear();
    cached_qpoints.clear();
    cached_point_indices.clear();

    const double       tolerance = 1e-10;
    const unsigned int n_points  = this->spatial_coordinates.size();

    // For each point, track which cell contains it and its global cell ID
    // We use global cell ID to ensure consistent assignment across ranks
    // (the cell with smallest global ID wins)
    std::vector<types::global_cell_index> point_to_cell_id(
      n_points, std::numeric_limits<types::global_cell_index>::max());
    std::vector<typename DoFHandler<dim>::active_cell_iterator> point_to_cell(
      n_points);
    std::vector<Point<dim>> point_to_unit(n_points);

    // For each locally owned cell, check which points are inside
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        const types::global_cell_index cell_id =
          cell->global_active_cell_index();

        // Get cell bounding box for quick rejection
        // Extend bbox generously for curved cells (quadratic mapping can
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

            // Quick bounding box check
            if (!bbox.point_inside(point))
              continue;

            // Detailed check: transform to unit coordinates
            try
              {
                const Point<dim> unit_point =
                  mapping.transform_real_to_unit_cell(cell, point);

                // Check if point is inside reference cell
                if (GeometryInfo<dim>::is_inside_unit_cell(unit_point,
                                                           tolerance))
                  {
                    // Use smallest cell ID to break ties consistently
                    if (cell_id < point_to_cell_id[k])
                      {
                        point_to_cell_id[k] = cell_id;
                        point_to_cell[k]    = cell;
                        point_to_unit[k]    = unit_point;
                      }
                  }
              }
            catch (...)
              {
                // Point not in this cell (transformation failed)
              }
          }
      }

    // Now find the global minimum cell ID for each point across all ranks
    std::vector<types::global_cell_index> global_min_cell_id(n_points);
    MPI_Allreduce(point_to_cell_id.data(),
                  global_min_cell_id.data(),
                  n_points,
                  MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN,
                  MPI_COMM_WORLD);

    // Build map: only keep points where this rank has the winning cell
    std::map<typename DoFHandler<dim>::active_cell_iterator,
             std::pair<std::vector<Point<dim>>, std::vector<unsigned int>>>
      cell_point_map;

    for (unsigned int k = 0; k < n_points; ++k)
      {
        // This rank wins if its cell ID matches the global minimum
        if (point_to_cell_id[k] == global_min_cell_id[k] &&
            point_to_cell_id[k] !=
              std::numeric_limits<types::global_cell_index>::max())
          {
            cell_point_map[point_to_cell[k]].first.push_back(point_to_unit[k]);
            cell_point_map[point_to_cell[k]].second.push_back(k);
          }
      }

    // Convert map to vectors for efficient iteration
    for (const auto &[cell, data] : cell_point_map)
      {
        cached_cells.push_back(cell);
        cached_qpoints.push_back(data.first);
        cached_point_indices.push_back(data.second);
      }

    // Diagnostic: count points assigned
    unsigned int local_points_found = 0;
    for (const auto &indices : cached_point_indices)
      local_points_found += indices.size();

    const unsigned int global_points_found =
      Utilities::MPI::sum(local_points_found, MPI_COMM_WORLD);

    this->pcout << "Point evaluation cache: " << global_points_found << " / "
                << n_points << " points assigned globally." << std::endl;

    if (global_points_found != n_points)
      this->pcout << "WARNING: " << (n_points - global_points_found)
                  << " points were not found in any cell!" << std::endl;
  }

  // ==========================================================================
  // Helper: Compute average cell diameter across all MPI ranks
  // ==========================================================================
  template <int dim>
  double
  DarcyAdjoint<dim>::compute_average_cell_diameter() const
  {
    double       local_diameter_sum = 0.0;
    unsigned int local_n_cells      = 0;

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          local_diameter_sum += cell->diameter();
          ++local_n_cells;
        }

    const double global_diameter_sum =
      Utilities::MPI::sum(local_diameter_sum, MPI_COMM_WORLD);
    const unsigned int global_n_cells =
      Utilities::MPI::sum(local_n_cells, MPI_COMM_WORLD);

    return global_diameter_sum / global_n_cells;
  }

  // ==========================================================================
  // Helper: Check if cell is near any observation point
  // ==========================================================================
  template <int dim>
  bool
  DarcyAdjoint<dim>::is_cell_near_any_observation(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const double                                          cutoff) const
  {
    const Point<dim> cell_center   = cell->center();
    const double     cell_diameter = cell->diameter();
    const double     threshold     = cutoff + cell_diameter;

    for (const auto &obs_point : this->spatial_coordinates)
      if (cell_center.distance(obs_point) < threshold)
        return true;

    return false;
  }

  // ==========================================================================
  // Mollified adjoint RHS assembly
  // ==========================================================================
  // Instead of using Dirac delta point observations (which cause mesh-dependent
  // spikes), we use a Gaussian mollification kernel:
  //
  //   RHS_i = sum_k (dL/dy_k) * integral[ G(x - x_k) * phi_i(x) dx ]
  //
  // where G(r) = exp(-|r|^2 / (2*sigma^2)) is an unnormalized Gaussian.
  //
  // Benefits:
  //   - Lower magnitude spikes (load spread over multiple elements)
  //   - Mesh-independent behavior (spikes don't sharpen with refinement)
  //   - Smoother gradients for the inverse problem
  // ==========================================================================

  template <int dim>
  void
  DarcyAdjoint<dim>::overwrite_adjoint_rhs()
  {
    TimerOutput::Scope timing_section(this->computing_timer,
                                      "Overwrite adjoint rhs");
    this->pcout << "Overwriting adjoint rhs with upstream gradient..."
                << std::endl;

    // -------------------------------------------------------------------------
    // Compute mollification parameters (adaptive based on mesh size)
    // -------------------------------------------------------------------------
    const double avg_cell_diameter = compute_average_cell_diameter();
    const double sigma_factor      = this->params.mollification_sigma_factor;
    const double sigma             = sigma_factor * avg_cell_diameter;
    const double sigma_sq          = sigma * sigma;
    const double cutoff            = 4.0 * sigma;
    const double cutoff_sq         = cutoff * cutoff;
    const double normalization =
      std::pow(2.0 * numbers::PI * sigma_sq, static_cast<double>(dim) / 2.0);

    this->pcout << "  Mollification: sigma = " << sigma << " (" << sigma_factor
                << " x avg_h = " << avg_cell_diameter << ")" << std::endl;

    // -------------------------------------------------------------------------
    // FEM setup
    // -------------------------------------------------------------------------
    const MappingQ<dim> mapping(2);
    const QGauss<dim>   quadrature(this->degree_u + 1);

    FEValues<dim> fe_values(mapping,
                            this->fe,
                            quadrature,
                            update_values | update_quadrature_points |
                              update_JxW_values);

    const unsigned int dofs_per_cell = this->fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature.size();
    const unsigned int n_obs_points  = this->spatial_coordinates.size();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    Vector<double>                       local_rhs(dofs_per_cell);

    // -------------------------------------------------------------------------
    // Identify velocity DoFs (exclude pressure DoFs from RHS assembly)
    // -------------------------------------------------------------------------
    std::vector<unsigned int> velocity_dof_indices;
    std::vector<unsigned int> velocity_components;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int comp = this->fe.system_to_component_index(i).first;
        if (comp < dim) // velocity components are 0, 1, 2 (in 3D)
          {
            velocity_dof_indices.push_back(i);
            velocity_components.push_back(comp);
          }
      }
    const unsigned int n_velocity_dofs = velocity_dof_indices.size();

    // -------------------------------------------------------------------------
    // Initialize RHS to zero
    // -------------------------------------------------------------------------
    this->system_rhs = 0;

    // -------------------------------------------------------------------------
    // Assembly loop over locally owned cells
    // -------------------------------------------------------------------------
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      {
        if (!cell->is_locally_owned())
          continue;

        // Quick rejection: skip cells far from all observation points
        if (!is_cell_near_any_observation(cell, cutoff))
          continue;

        // Initialize FE values and local data
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        local_rhs = 0;

        // Loop over quadrature points
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const Point<dim> &x_q = fe_values.quadrature_point(q);
            const double      JxW = fe_values.JxW(q);

            // Accumulate contributions from nearby observation points
            for (unsigned int k = 0; k < n_obs_points; ++k)
              {
                const Point<dim> &x_obs   = this->spatial_coordinates[k];
                const double      dist_sq = (x_q - x_obs).norm_square();

                // Skip observation points outside the cutoff radius
                if (dist_sq > cutoff_sq)
                  continue;

                // Evaluate Gaussian kernel (unnormalized)
                const double gaussian = std::exp(-dist_sq / (2.0 * sigma_sq));

                // Add contribution to velocity DoFs only
                for (unsigned int v = 0; v < n_velocity_dofs; ++v)
                  {
                    const unsigned int i    = velocity_dof_indices[v];
                    const unsigned int comp = velocity_components[v];
                    const double       phi  = fe_values.shape_value(i, q);

                    local_rhs(i) += data_vec[k][comp] * gaussian * phi * JxW;
                  }
              }
          }

        // Accumulate local contributions to global RHS
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          this->system_rhs(local_dof_indices[i]) += local_rhs(i);
      }

    // -------------------------------------------------------------------------
    // Finalize: compress and normalize
    // -------------------------------------------------------------------------
    this->system_rhs.compress(VectorOperation::add);
    this->system_rhs /= normalization;

    this->pcout << "  Adjoint RHS assembled successfully." << std::endl;
    this->pcout << "  RHS L2 norm: " << this->system_rhs.l2_norm() << std::endl;
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::run(const Parameters &params)
  {
    this->params = params;
    run_simulation();

    this->pcout << "Adjoint problem solved successfully!" << std::endl;
    this->computing_timer.print_summary();
    this->computing_timer.reset();

    this->pcout << std::endl;
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::create_rf_laplace_operator()
  {
    TimerOutput::Scope timing_section(this->computing_timer,
                                      "Create rf laplace operator");
    this->pcout << "Creating random field laplace operator..." << std::endl;

    const IndexSet locally_owned = this->rf_dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant =
      DoFTools::extract_locally_relevant_dofs(this->rf_dof_handler);
    AffineConstraints<double> rf_constraints;
    rf_constraints.clear();
    DoFTools::make_hanging_node_constraints(this->rf_dof_handler,
                                            rf_constraints);
    rf_constraints.close();

    TrilinosWrappers::SparsityPattern sp_rf(locally_owned,
                                            locally_owned,
                                            locally_relevant,
                                            MPI_COMM_WORLD);

    DoFTools::make_sparsity_pattern(this->rf_dof_handler,
                                    sp_rf,
                                    rf_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    sp_rf.compress();

    this->rf_laplace_matrix.reinit(sp_rf);

    this->mean_rf.reinit(locally_owned, MPI_COMM_WORLD);
    this->mean_rf = 0;

    const QGauss<dim> quadrature(this->degree_u + 1);
    MappingQ<dim>     mapping(2);

    const Function<dim, double> *coefficient = nullptr;
    MatrixCreator::create_laplace_matrix(mapping,
                                         this->rf_dof_handler,
                                         quadrature,
                                         this->rf_laplace_matrix,
                                         coefficient,
                                         rf_constraints);

    TrilinosWrappers::SparseMatrix rf_mass_matrix(sp_rf);

    MatrixCreator::create_mass_matrix(mapping,
                                      this->rf_dof_handler,
                                      quadrature,
                                      rf_mass_matrix,
                                      coefficient,
                                      rf_constraints);

    const double epsilon = 1e-5;
    this->rf_laplace_matrix.add(epsilon, rf_mass_matrix);
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::add_prior_gradient_to_adjoint()
  {
    const double a = 1e-9 + this->rf_dof_handler.n_dofs() / 2.0;

    const IndexSet &owned = this->rf_dof_handler.locally_owned_dofs();
    TrilinosWrappers::MPI::Vector x_minus_mean, prior_grad;
    x_minus_mean.reinit(owned, MPI_COMM_WORLD);
    prior_grad.reinit(owned, MPI_COMM_WORLD);

    for (const auto idx : owned)
      x_minus_mean[idx] = this->x_vec_distributed[idx];
    x_minus_mean.compress(VectorOperation::insert);

    x_minus_mean.add(-1.0, this->mean_rf);

    this->rf_laplace_matrix.vmult(prior_grad, x_minus_mean);

    const double b_0       = 1e-9;
    const double quad_form = x_minus_mean * prior_grad;
    const double b         = b_0 + 0.5 * quad_form;

    prior_grad *= -(a / b);

    for (const auto idx : owned)
      grad_log_lik_x_distributed[idx] += prior_grad[idx];

    Utilities::MPI::sum(grad_log_lik_x_distributed,
                        MPI_COMM_WORLD,
                        grad_log_lik_x);

    this->pcout << "Successfully added prior gradient to adjoint gradient!"
                << std::endl;
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::output_gradient_pvtu()
  {
    TimerOutput::Scope timing_section(this->computing_timer,
                                      "Output gradient VTU");
    this->pcout << "Writing gradient to VTU file..." << std::endl;

    const IndexSet &locally_owned = this->rf_dof_handler.locally_owned_dofs();
    const IndexSet  locally_relevant =
      DoFTools::extract_locally_relevant_dofs(this->rf_dof_handler);

    TrilinosWrappers::MPI::Vector gradient_owned(locally_owned, MPI_COMM_WORLD);
    TrilinosWrappers::MPI::Vector gradient_distributed(locally_owned,
                                                       locally_relevant,
                                                       MPI_COMM_WORLD);

    for (const auto idx : locally_owned)
      gradient_owned[idx] = grad_log_lik_x[idx];

    gradient_distributed = gradient_owned;

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->rf_dof_handler);
    data_out.add_data_vector(gradient_distributed, "gradient");

    MappingQ<dim> mapping(2);
    data_out.build_patches(mapping, 2, DataOut<dim>::curved_inner_cells);

    const std::string filename      = this->params.output_prefix + "gradient";
    const std::string stripped_path = this->params.output_directory + "/";

    data_out.write_vtu_with_pvtu_record(
      stripped_path, filename, 0, MPI_COMM_WORLD, 1, 1);

    this->pcout << "Gradient written to " << stripped_path << filename
                << ".pvtu" << std::endl;
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::run_simulation()
  {
    const bool adjoint_solve = true;

    this->setup_grid_and_dofs();

    // Construct full path to adjoint data file
    // (assumed to be in same directory as input npy file)
    std::filesystem::path input_path(this->params.input_npy_file);
    std::filesystem::path adjoint_data_path =
      input_path.parent_path() / this->params.adjoint_data_file;

    const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler, {0, 0, 0, 1});
    const types::global_dof_index n_u = dofs_per_block[0],
                                  n_p = dofs_per_block[1];
    std::vector<IndexSet> partitioning;
    IndexSet              index_set = this->dof_handler.locally_owned_dofs();
    partitioning.push_back(index_set.get_view(0, n_u));
    partitioning.push_back(index_set.get_view(n_u, n_u + n_p));

    IndexSet relevant_set =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);
    std::vector<IndexSet> relevant_partitioning;
    relevant_partitioning.push_back(relevant_set.get_view(0, n_u));
    relevant_partitioning.push_back(relevant_set.get_view(n_u, n_u + n_p));

    solution_primary_problem.reinit(partitioning, MPI_COMM_WORLD);
    solution_primary_distributed.reinit(partitioning,
                                        relevant_partitioning,
                                        MPI_COMM_WORLD);

    this->read_input_npy();
    this->generate_coordinates();
    setup_point_evaluation_cache(); // Cache RPE for efficient RHS assembly
    read_upstream_gradient_npy(adjoint_data_path.string());
    read_primary_solution();
    this->assemble_approx_schur_complement();
    this->assemble_system();
    overwrite_adjoint_rhs();
    this->solve(adjoint_solve);
    final_inner_adjoint_product();
    create_rf_laplace_operator();
    add_prior_gradient_to_adjoint();
    write_gradient_to_npy();
    output_gradient_pvtu();
    write_adjoint_solution_pvtu();
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::write_gradient_to_npy()
  {
    unsigned int rows    = grad_log_lik_x.size();
    unsigned int columns = 1;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        const std::string filename = this->params.output_directory + "/" +
                                     this->params.output_prefix +
                                     "grad_solution.npy";
        this->write_data_to_npy(filename, grad_log_lik_x, rows, columns);
      }
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::write_adjoint_solution_pvtu()
  {
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
    data_out.add_data_vector(this->dof_handler,
                             this->solution_distributed,
                             solution_names,
                             interpretation);

    MappingQ<dim> mapping(2);
    data_out.build_patches(mapping,
                           this->degree_u,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename =
      this->params.output_prefix + "adjoint_solution";
    const std::string stripped_path = this->params.output_directory + "/";

    constexpr unsigned int n_digits_counter = 2;
    constexpr unsigned int num_vtu_files    = 1;
    constexpr unsigned int cycle            = 0;
    data_out.write_vtu_with_pvtu_record(stripped_path,
                                        filename,
                                        cycle,
                                        MPI_COMM_WORLD,
                                        n_digits_counter,
                                        num_vtu_files);

    this->pcout << "DEBUG: Wrote adjoint solution to " << stripped_path
                << filename << ".pvtu" << std::endl;
  }

  template <int dim>
  void
  DarcyAdjoint<dim>::final_inner_adjoint_product()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "Final inner adjoint product");
    this->pcout << "Final inner adjoint product with jacobi_k_mat_inv"
                << std::endl;

    unsigned int x_dim = this->rf_dof_handler.n_dofs();

    grad_log_lik_x.assign(x_dim, 0.0);
    grad_log_lik_x_distributed.assign(x_dim, 0.0);
    this->x_vec_distributed = this->x_vec; // Ensure distributed x_vec is set


    const MappingQ<dim> mapping(2);

    const QGauss<dim> quadrature(this->degree_u + 1);

    FEValues<dim>      fe_values(mapping,
                            this->fe,
                            quadrature,
                            update_values | update_JxW_values);
    FEValues<dim>      fe_rf_values(mapping,
                               this->rf_fe_system,
                               quadrature,
                               update_values);
    const unsigned int n_q_points = fe_values.n_quadrature_points;

    FEValuesExtractors::Vector velocities(0);
    const unsigned int         dofs_per_cell = this->fe.n_dofs_per_cell();
    const unsigned int rf_dofs_per_cell = this->rf_fe_system.n_dofs_per_cell();

    std::vector<types::global_dof_index> local_rf_dof_indices(rf_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double>         solution_local(dofs_per_cell);
    std::vector<double>         solution_primary_local(dofs_per_cell);
    std::vector<double>         rf_value(n_q_points);
    std::vector<Tensor<1, dim>> velocity_adjoint_q(n_q_points);
    std::vector<Tensor<1, dim>> velocity_primary_q(n_q_points);
    std::vector<double>         local_gradient(rf_dofs_per_cell);

    this->solution_distributed   = this->solution;
    solution_primary_distributed = solution_primary_problem;

    for (const auto &cell_tria : this->triangulation.active_cell_iterators())
      {
        const auto &cell =
          cell_tria->as_dof_handler_iterator(this->dof_handler);
        const auto &rf_cell =
          cell_tria->as_dof_handler_iterator(this->rf_dof_handler);

        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            fe_rf_values.reinit(rf_cell);

            cell->get_dof_indices(local_dof_indices);
            rf_cell->get_dof_indices(local_rf_dof_indices);

            fe_rf_values.get_function_values(this->x_vec_distributed, rf_value);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                solution_local[i] =
                  this->solution_distributed[local_dof_indices[i]];
                solution_primary_local[i] =
                  solution_primary_distributed[local_dof_indices[i]];
              }

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

            std::fill(local_gradient.begin(), local_gradient.end(), 0.0);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                const double JxW_q = fe_values.JxW(q);

                const double exp_neg_rf = 1.0 / std::exp(rf_value[q]);
                const double vel_dot =
                  velocity_adjoint_q[q] * velocity_primary_q[q];
                const double common_factor = JxW_q * exp_neg_rf * vel_dot;

                for (unsigned int k = 0; k < rf_dofs_per_cell; ++k)
                  {
                    const double phi_k = fe_rf_values.shape_value(k, q);
                    local_gradient[k] += common_factor * phi_k;
                  }
              }

            for (unsigned int k = 0; k < rf_dofs_per_cell; ++k)
              grad_log_lik_x_distributed[local_rf_dof_indices[k]] +=
                local_gradient[k];
          }
      }
    this->pcout << "grad_log_x (distributed) successfully assembled!"
                << std::endl;
  }

} // namespace darcy

#endif // DARCY_ADJOINT_H
