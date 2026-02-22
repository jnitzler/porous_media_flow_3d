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
The executables use deal.II's `ParameterHandler` to read configuration from a JSON or PRM file.

### Parameters

| Parameter | Section | Default | Description |
|---|---|---|---|
| `pressure fe degree` | Discretization | 1 | Polynomial degree for pressure FE (velocity = degree + 1) |
| `random field fe degree` | Discretization | 2 | Polynomial degree for random field FE |
| `refinement level` | Discretization | 4 | Global mesh refinement level |
| `refinement level obs` | Discretization | 3 | Observation mesh refinement level |
| `alpha` | Prior | 2 | SPDE operator power $n$: Matérn smoothness $\nu = n - d/2$. Use $n \geq 2$ for 3D |
| `kappa squared` | Prior | 16.0 | Controls prior correlation length: $\rho = \sqrt{8\nu}/\kappa$ |
| `input npy file` | Input/Output | — | Path to random field coefficients (.npy) |
| `output directory` | Input/Output | `output` | Results directory |
| `output prefix` | Input/Output | `""` | Filename prefix for outputs |
| `adjoint data file` | Input/Output | `adjoint_data.npy` | Upstream gradient filename (adjoint only) |

### SPDE prior parameters

The adjoint solver uses an SPDE--GMRF prior (Lindgren, Rue & Lindström, 2011) with precision matrix $Q = z \, B_n^T M^{-1} B_n$, where $B_n = A_\kappa (M^{-1} A_\kappa)^{n-1}$ and $A_\kappa = \kappa^2 M + G$. The key parameters are:
- **`alpha`** ($n$): SPDE operator power. Determines Matérn smoothness $\nu = n - d/2$. In 3D, $n=2$ gives $\nu=1/2$ (continuous paths), $n=3$ gives $\nu=3/2$ (differentiable). Must be $\geq 2$ for valid 3D fields.
- **`kappa squared`** ($\kappa^2$): Controls the practical correlation length $\rho = \sqrt{8\nu}/\kappa$. Examples for $n=2$ ($\nu=0.5$) on the donut domain (outer radius 1.0):

  | $\kappa^2$ | $\rho$ | Interpretation |
  |---|---|---|
  | 4.0 | 1.0 | Full domain extent |
  | 16.0 | 0.5 | Half the outer radius |
  | 64.0 | 0.25 | Local features |

### Example JSON configuration

```json
{
  "Discretization": {
    "pressure fe degree": 1,
    "random field fe degree": 2,
    "refinement level": 4,
    "refinement level obs": 3
  },
  "Prior": {
    "alpha": 2,
    "kappa squared": 16.0
  },
  "Input/Output": {
    "input npy file": "input/markov_field_5.npy",
    "output directory": "output",
    "output prefix": "my_sim_",
    "adjoint data file": "adjoint_data.npy"
  }
}
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
