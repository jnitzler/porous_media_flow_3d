// parameters.h
// Parameter handling for Darcy flow simulations using deal.II ParameterHandler

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

#include <filesystem>
#include <string>

namespace darcy
{
  // ===========================================================================
  // Parameters struct: Handles input/output configuration
  // ===========================================================================
  struct Parameters
  {
    std::string input_npy_file;
    std::string output_directory;
    std::string output_prefix;
    std::string adjoint_data_file;
    unsigned int fe_degree;

    // Declare all parameters in the ParameterHandler
    static void
    declare_parameters(dealii::ParameterHandler &prm);

    // Parse parameters from the ParameterHandler
    void
    parse_parameters(dealii::ParameterHandler &prm);
  };

  // ===========================================================================
  // Implementation
  // ===========================================================================

  void
  Parameters::declare_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      prm.declare_entry("pressure fe degree",
                        "1",
                        dealii::Patterns::Integer(0),
                        "Polynomial degree of the pressure finite element");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input/Output");
    {
      prm.declare_entry("input npy file",
                        "input/markov_field_5.npy",
                        dealii::Patterns::FileName(),
                        "Path to the input npy file containing random field "
                        "coefficients");

      prm.declare_entry("output directory",
                        "output",
                        dealii::Patterns::Anything(),
                        "Path to the output directory for results");

      prm.declare_entry("output prefix",
                        "",
                        dealii::Patterns::Anything(),
                        "Prefix for output filenames (e.g., 'run1_' or 'test_')");

      prm.declare_entry(
        "adjoint data file",
        "adjoint_data.npy",
        dealii::Patterns::FileName(),
        "Filename of the adjoint data npy file (assumed to be in the same "
        "directory as the input npy file, only used for adjoint problem)");
    }
    prm.leave_subsection();
  }

  void
  Parameters::parse_parameters(dealii::ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      fe_degree = prm.get_integer("pressure fe degree");
    }
    prm.leave_subsection();

    prm.enter_subsection("Input/Output");
    {
      input_npy_file     = prm.get("input npy file");
      output_directory   = prm.get("output directory");
      output_prefix      = prm.get("output prefix");
      adjoint_data_file  = prm.get("adjoint data file");
    }
    prm.leave_subsection();
  }

} // namespace darcy

#endif // PARAMETERS_H
