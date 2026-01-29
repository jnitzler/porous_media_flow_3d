# Isotropic Darcy flow

This is a fully parallelized implementation of a Darcy flow PDE with a transverse isotropic permeability field, using deal.II. The governing equations that are solved read as follows:

$$
\begin{align*}
K^{-1} \boldsymbol{u} + \nabla p &= 0 \quad \text{in } \Omega \\
-\text{div } \boldsymbol{u} &= f \quad \text{in } \Omega \\
p &= g \quad \text{on } \partial\Omega
\end{align*}
$$

With $K$ being a $\dim \times \dim$ permeability tensor, $\boldsymbol{u}$ the flow velocity and $p$ the pressure. We implemented a transverse isotropic permeability field $K(\boldsymbol{x})$, which is modeled by random fields, parameterized by a set of coefficients $\boldsymbol{x}$.

The project builds two executables:
1. `darcy_forward.cc` → `darcy_forward`: The main executable and forward solve of the PDE for a specific choice of random field coefficients $\boldsymbol{x}$ which imposes the mapping: $\boldsymbol{y} = f(\boldsymbol{x})$
2. `darcy_adjoint.cc` → `darcy_adjoint`: The associated adjoint problem that returns the derivative $\frac{\partial g(f(\boldsymbol{x}))}{\partial \boldsymbol{x}}$ for an objective function $g$

## Random permeability field
The random isotropic permeability tensor $K(\boldsymbol{x})$ is modeled as follows:

$$K(\boldsymbol{x}) = \exp(\boldsymbol{x}) \cdot I$$

such that $\boldsymbol{x}$ can be inferred without constraints.

## Configuration
The executables use deal.II's `ParameterHandler` to read configuration from a JSON or PRM file. The parameter file specifies:
- `pressure fe degree`: Polynomial degree of the finite element basis functions for pressure (e.g., 1 for linear elements, 2 for quadratic)
- `refinement level`: Global refinement level for the main triangulation (default: 4)
- `refinement level obs`: Global refinement level for the observation triangulation (default: 3)
- `input npy file`: Path to the input npy file containing random field coefficients
- `output directory`: Path to the output directory for results
- `output prefix`: Prefix for output filenames (e.g., 'run1_' or 'test_')
- `adjoint data file`: Filename of the adjoint data npy file (assumed to be in the same directory as the input npy file, only used for adjoint problem)
- `mollification sigma factor`: Factor to scale the mollification sigma for the adjoint problem's right-hand side (smoothing of the delta distributions at observation points)

Example JSON configuration (`parameters.json`):
```json
{
  "Discretization": {
    "pressure fe degree": 1,
    "refinement level": 4,
    "refinement level obs": 3
  },
  "Input/Output": {
    "input npy file": "input/markov_field_5.npy",
    "output directory": "output",
    "output prefix": "my_sim_",
    "adjoint data file": "adjoint_data.npy"
  },
  "Adjoint": {
    "mollification sigma factor": 1.0
  }
}
```

Alternatively, you can use the PRM format (`parameters.prm`):
```prm
subsection Discretization
  set pressure fe degree = 1
  set refinement level = 4
  set refinement level obs = 3
end

subsection Input/Output
  set input npy file = input/markov_field_5.npy
  set output directory = output
  set output prefix = 
  set adjoint data file = adjoint_data.npy
end

subsection Adjoint
  set mollification sigma factor = 1.0
end
```

## Running the executables
The primary problem can be started with:
```bash
mpirun -np <num_procs> darcy_forward parameters.json
```

The associated adjoint problem:
```bash
mpirun -np <num_procs> darcy_adjoint parameters.json
```

Note: The `adjoint_data.npy` file should be located in the same directory as the input npy file specified in the parameter file.

## Setup, installation and dependencies

This code requires the installation and setup of [deal.II](https://www.dealii.org/), furthermore, the [Trilinos](https://trilinos.github.io/) project needs to be configured and installed. For parallel computing, respectively partitioning we furthermore require the installation of 
[p4est](pymc.io/projects/examples/en/latest/gallery.html). 

## Associated deal.II tutorials

The code is largely based on the deal.II [tutorial 20](https://www.dealii.org/developer/doxygen/deal.II/step_20.html) for the overall idea, [tutorials 21](https://www.dealii.org/developer/doxygen/deal.II/step_21.html) and [22](https://www.dealii.org/developer/doxygen/deal.II/step_22.html) for handling block systems and further enhanced solution strategies. We furthermore used [tutorial 43](https://www.dealii.org/developer/doxygen/deal.II/step_43.html) for improved solvers and an efficient block-preconditioner. Finally, [tutorials 31](https://www.dealii.org/developer/doxygen/deal.II/step_31.html) and [32](https://www.dealii.org/developer/doxygen/deal.II/step_32.html) were used to parallelize the code for use on multiple processors.
