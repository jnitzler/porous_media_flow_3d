# Isotropic Darcy flow

This is a fully parallelized implementation of a Darcy flow PDE with a transverse isotropic permeability field, using deal.II. The governing equations that are solved read as follows:

$$
\begin{align*}
K^{-1} \boldsymbol{u} + \nabla p &= 0 \quad \text{in } \Omega \\
-\text{div} \, \boldsymbol{u} &= f \quad \text{in } \Omega \\
p &= g \quad \text{on } \partial\Omega
\end{align*}
$$

With $K$ being a $\dim \times \dim$ permeability tensor, $\boldsymbol{u}$ the flow velocity and $p$ the pressure. We implemented a transverse isotropic permeability field $K(\boldsymbol{x})$, which is modeled by random fields, parameterized by a set of coefficients $\boldsymbol{x}$.

The project builds two executables:
1. `darcy.cc` → `darcy`: The main executable and forward solve of the PDE for a specific choice of random field coefficients $\boldsymbol{x}$ which imposes the mapping: $\boldsymbol{y} = f(\boldsymbol{x})$
2. `darcy_adjoint.cc` → `darcy_adjoint`: The associated adjoint problem that returns the derivative $\frac{\partial g(f(\boldsymbol{x}))}{\partial \boldsymbol{x}}$ for an objective function $g$

## Random permeability field
The random isotropic permeability tensor $K(\boldsymbol{x})$ is modeled as follows:

$$K(\boldsymbol{x}) = \exp(\boldsymbol{x}) \cdot I$$

such that $\boldsymbol{x}$ can be inferred without constraints.
## Running the executables
The executables require
- a path to an input file
- a path to the output directory
- the adjoint executable requires additionally a file `adjoint_data.npy` which has to be located in the same directory as the input file.

The primary problem can be started with `mpirun -np <num_procs> darcy <path/to/test_input.npy> <path/to/output_dir>`. The associated adjoint problem with analogous with `mpirun -np <num_procs> darcy_adjoint <path/to/test_input.npy> <path/to/output_dir>`

## Setup, installation and dependencies
This code requires the installation and setup of [deal.ii](https://www.dealii.org/), furthermore, the [Trilinos](https://trilinos.github.io/) project needs to be configured and installed. For parallel computing, respectively partitioning we furthermore require the installation of 
[p4est](pymc.io/projects/examples/en/latest/gallery.html). The implementation of the [random fields is a separate project](https://gitlab.lrz.de/adjoint_problems/random_fields) that needs to be downloaded. The import path for the respective header file might need to be adjusted in the source code.
## Associated deal.ii tutorials
The code is largely based on the deal.ii [tutorial 20](https://www.dealii.org/developer/doxygen/deal.II/step_20.html) for the overall idea, [tutorials 21](https://www.dealii.org/developer/doxygen/deal.II/step_21.html) and [22](https://www.dealii.org/developer/doxygen/deal.II/step_22.html) for handling block systems and further enhanced solution strategies. We furthermore used [tutorial 43](https://www.dealii.org/developer/doxygen/deal.II/step_43.html) for improved solvers and an efficient block-preconditioner. Finally, [tutorials 31](https://www.dealii.org/developer/doxygen/deal.II/step_31.html) and [32](https://www.dealii.org/developer/doxygen/deal.II/step_32.html) were used to parallelize the code for use on multiple processors.
