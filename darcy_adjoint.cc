#include "darcy_adjoint.h"
#include "parameters.h"

#include <filesystem>

// Explicit template instantiation
template class darcy::DarcyAdjoint<3>;

// ---------------------- main function adjoint -----------------------------
int
main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace darcy;

      // Check for correct usage
      if (argc != 2)
        {
          std::cerr << "Usage: " << argv[0] << " <parameter_file.json>"
                    << std::endl;
          std::cerr << "Example: mpirun -np 4 " << argv[0]
                    << " parameters.json" << std::endl;
          return 1;
        }

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      // Setup parameter handler
      ParameterHandler prm;
      Parameters       params;
      Parameters::declare_parameters(prm);

      // Parse JSON/PRM file
      try
        {
          prm.parse_input(argv[1]);
        }
      catch (const std::exception &exc)
        {
          std::cerr << "Error parsing parameter file: " << argv[1]
                    << std::endl;
          std::cerr << exc.what() << std::endl;
          return 1;
        }

      params.parse_parameters(prm);

      // Create output directory if it doesn't exist
      std::filesystem::create_directories(params.output_directory);

      // Run adjoint solver
      DarcyAdjoint<3>    mixed_laplace_problem(params.fe_degree);
      mixed_laplace_problem.run(params);
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
