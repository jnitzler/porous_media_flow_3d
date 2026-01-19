// darcy_forward.h
// Forward Darcy flow solver class.
// Implements the forward simulation for the Darcy flow problem.

#ifndef DARCY_FORWARD_H
#define DARCY_FORWARD_H

#include "darcy_base.h"

namespace darcy
{
  // ===========================================================================
  // Darcy class: Forward Darcy flow solver
  // ===========================================================================
  template <int dim>
  class Darcy : public DarcyBase<dim>
  {
  public:
    // Constructor
    explicit Darcy(const unsigned int degree_p);

    // Main entry point for forward simulation
    void
    run(const std::string &input_path, const std::string &output_path) override;

  private:
    // -------------------------------------------------------------------------
    // Forward-specific output methods
    // -------------------------------------------------------------------------
    void
    output_full_velocity_npy(
      const std::string &output_path); // Full solution to .npy
    void
    output_velocity_at_observation_points_npy(const std::string &output_path);

    // -------------------------------------------------------------------------
    // Simulation driver
    // -------------------------------------------------------------------------
    void
    run_simulation(const std::string &input_path,
                   const std::string &output_path);
  };

  // Constructor implementation
  template <int dim>
  Darcy<dim>::Darcy(const unsigned int degree_p)
    : DarcyBase<dim>(degree_p)
  {}

} // namespace darcy

#endif // DARCY_FORWARD_H
