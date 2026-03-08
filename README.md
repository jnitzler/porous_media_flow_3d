# Isotropic Darcy flow

This is a fully parallelized implementation of a Darcy flow PDE with an isotropic permeability field on a 3D eccentric hyper-shell ("donut") geometry, using deal.II and Trilinos. The governing equations read:

$$
\begin{align*}
K^{-1} \boldsymbol{u} + \nabla p &= 0 \quad \text{in } \Omega \\
-\text{div } \boldsymbol{u} &= f \quad \text{in } \Omega \\
p &= g \quad \text{on } \partial\Omega
\end{align*}
$$

With $K$ being a $\dim \times \dim$ permeability tensor, $\boldsymbol{u}$ the flow velocity and $p$ the pressure. The isotropic permeability field $K(\boldsymbol{x}) = \exp(\boldsymbol{x}) \cdot I$ is parameterized by random field coefficients $\boldsymbol{x}$.

The project builds three executables:
1. `darcy_forward`: Forward solve $\boldsymbol{y} = f(\boldsymbol{x})$ for given random field coefficients
2. `darcy_adjoint`: Adjoint solve returning the gradient $\frac{\partial g(f(\boldsymbol{x}))}{\partial \boldsymbol{x}}$
3. `export_sparsity`: Exports the prior precision matrix sparsity pattern and values (COO format as `.npy` files). This is needed to construct a sparse variational approximation in the stochastic variational inference (SVI) framework — the FE mesh-induced sparsity pattern of $Q$ motivates the sparsity structure of the variational precision matrix, ensuring the approximation respects the local correlation structure of the prior.

Both the forward and adjoint executables support 2D and 3D via the `spatial dimension` parameter.

## Configuration

The executables use deal.II's `ParameterHandler` to read configuration from a JSON file.

### Parameters

| Parameter | Section | Default | Description |
|---|---|---|---|
| `spatial dimension` | Discretization | 3 | Spatial dimension (2 or 3) |
| `pressure fe degree` | Discretization | 1 | Polynomial degree for pressure FE (velocity = degree + 1) |
| `random field fe degree` | Discretization | 2 | Polynomial degree for random field FE |
| `refinement level` | Discretization | 4 | Global mesh refinement level |
| `refinement level obs` | Discretization | 3 | Observation mesh refinement level |
| `nugget` | Prior | 1e-6 | Nugget for Markov prior: $Q = G + \varepsilon M$ |
| `ground truth` | Input/Output | false | Use analytical reference field instead of reading from file |
| `input npy file` | Input/Output | — | Path to random field coefficients (.npy) |
| `output directory` | Input/Output | `output` | Results directory |
| `output prefix` | Input/Output | `""` | Filename prefix for outputs |
| `adjoint data file` | Input/Output | `adjoint_data.npy` | Upstream gradient filename (adjoint only) |

### Prior

The adjoint solver uses a Markov prior with precision matrix $Q = G + \varepsilon M$, where $G$ is the stiffness (Laplacian) matrix, $M$ the mass matrix, and $\varepsilon$ the nugget term. The precision scaling $z$ is updated adaptively via a conjugate Gamma hyper-prior (Student-t-like behavior):

$$z = \frac{a_0 + n/2}{b_0 + \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu})^T Q (\boldsymbol{x} - \boldsymbol{\mu})}$$

### Example JSON configuration

```json
{
  "Discretization": {
    "spatial dimension": 3,
    "pressure fe degree": 1,
    "random field fe degree": 2,
    "refinement level": 4,
    "refinement level obs": 3
  },
  "Prior": {
    "nugget": 1.0e-4
  },
  "Input/Output": {
    "ground truth": false,
    "input npy file": "input/coefficients.npy",
    "output directory": "output",
    "output prefix": "my_sim_",
    "adjoint data file": "adjoint_data.npy"
  }
}
```

## Running the executables

The forward problem:
```bash
mpirun -np <num_procs> darcy_forward parameters.json
```

The adjoint problem:
```bash
mpirun -np <num_procs> darcy_adjoint parameters.json
```

The adjoint requires the forward solution (`*_solution_full.npy`) and an upstream gradient file (`adjoint_data.npy` in the same directory as `input npy file`).

## Setup, installation and dependencies

This code requires:
- [deal.II](https://www.dealii.org/) >= 9.5.0 (with Trilinos and p4est enabled)
- [Trilinos](https://trilinos.github.io/)
- [p4est](https://www.p4est.org/) for parallel mesh partitioning
- MPI

### Building

The project uses out-of-source CMake builds:

```bash
mkdir -p build/release && cd build/release
cmake -DDEAL_II_DIR=/path/to/dealii-install -DCMAKE_BUILD_TYPE=Release ../..
make -j$(nproc)
```

## Associated deal.II tutorials

The code is largely based on the deal.II [tutorial 20](https://www.dealii.org/developer/doxygen/deal.II/step_20.html) for the overall idea, [tutorials 21](https://www.dealii.org/developer/doxygen/deal.II/step_21.html) and [22](https://www.dealii.org/developer/doxygen/deal.II/step_22.html) for handling block systems and further enhanced solution strategies. We furthermore used [tutorial 43](https://www.dealii.org/developer/doxygen/deal.II/step_43.html) for improved solvers and an efficient block-preconditioner. Finally, [tutorials 31](https://www.dealii.org/developer/doxygen/deal.II/step_31.html) and [32](https://www.dealii.org/developer/doxygen/deal.II/step_32.html) were used to parallelize the code for use on multiple processors.
