// parameters.h
// Parameter handling for Darcy flow simulations using deal.II ParameterHandler

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>
#include <filesystem>
#include <string>

using namespace dealii;

namespace darcy
{
  // ===========================================================================
  // Parameters struct: Handles input/output configuration
  // ===========================================================================
  struct Parameters
  {
    // Discretization
    unsigned int fe_degree;
    unsigned int degree_rf;
    unsigned int refinement_level;
    unsigned int refinement_level_obs;

    // Input/Output
    std::string input_npy_file;
    std::string output_directory;
    std::string output_prefix;
    std::string adjoint_data_file;

    // Adjoint solver settings
    double
      mollification_sigma_factor; // Gaussian width = factor * avg cell diameter

    // Declare all parameters in the ParameterHandler
    static void
    declare_parameters(ParameterHandler &prm);

    // Parse parameters from the ParameterHandler
    void
    parse_parameters(ParameterHandler &prm);
  };

  // ===========================================================================
  // Implementation
  // ===========================================================================

  void
  Parameters::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      prm.declare_entry("pressure fe degree",
                        "1",
                        Patterns::Integer(0),
                        "Polynomial degree of the pressure finite element");

      prm.declare_entry("random field fe degree",
                        "2",
                        Patterns::Integer(0),
                        "Polynomial degree of the random field finite element");

      prm.declare_entry("refinement level",
                        "4",
                        Patterns::Integer(0),
                        "Global refinement level for the main triangulation");

      prm.declare_entry(
        "refinement level obs",
        "3",
        Patterns::Integer(0),
        "Global refinement level for the observation triangulation");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input/Output");
    {
      prm.declare_entry("input npy file",
                        "input/markov_field_5.npy",
                        Patterns::FileName(),
                        "Path to the input npy file containing random field "
                        "coefficients");

      prm.declare_entry("output directory",
                        "output",
                        Patterns::Anything(),
                        "Path to the output directory for results");

      prm.declare_entry(
        "output prefix",
        "",
        Patterns::Anything(),
        "Prefix for output filenames (e.g., 'run1_' or 'test_')");

      prm.declare_entry(
        "adjoint data file",
        "adjoint_data.npy",
        Patterns::FileName(),
        "Filename of the adjoint data npy file (assumed to be in the same "
        "directory as the input npy file, only used for adjoint problem)");
    }
    prm.leave_subsection();

    prm.enter_subsection("Adjoint");
    {
      prm.declare_entry(
        "mollification sigma factor",
        "1.5",
        Patterns::Double(0.1, 10.0),
        "Width of Gaussian mollification kernel as multiple of average cell "
        "diameter. Recommended: 1.0-3.0. Smaller values give sharper point "
        "loads, larger values give smoother gradients.");
    }
    prm.leave_subsection();
  }

  void
  Parameters::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      fe_degree            = prm.get_integer("pressure fe degree");
      degree_rf            = prm.get_integer("random field fe degree");
      refinement_level     = prm.get_integer("refinement level");
      refinement_level_obs = prm.get_integer("refinement level obs");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input/Output");
    {
      input_npy_file    = prm.get("input npy file");
      output_directory  = prm.get("output directory");
      output_prefix     = prm.get("output prefix");
      adjoint_data_file = prm.get("adjoint data file");
    }
    prm.leave_subsection();

    prm.enter_subsection("Adjoint");
    {
      mollification_sigma_factor = prm.get_double("mollification sigma factor");
    }
    prm.leave_subsection();
  }

} // namespace darcy

#endif // PARAMETERS_H
