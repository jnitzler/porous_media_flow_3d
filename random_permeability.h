// random_permeability.h
// Permeability field definitions for the Darcy solver.
//
// The permeability tensor is defined as K = exp(x) * I, where:
//   - x is the log-permeability (random field parameter)
//   - I is the identity tensor (isotropic permeability)
//
// This file provides:
//   - RefScalar: Analytical test function for log-permeability
//   - get_k_mat: Compute K from log-permeability value
//   - get_jacobi_inv_kmat: Derivative d(K^{-1})/dx for adjoint gradient

#ifndef RANDOM_PERMEABILITY_H
#define RANDOM_PERMEABILITY_H

#include <cmath>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <vector>

namespace darcy
{
  using namespace dealii;

  namespace RandomMedium
  {
    // =========================================================================
    // RefScalar: Analytical reference log-permeability field for testing.
    // Returns smooth oscillatory pattern with optional inclusion (3D).
    // =========================================================================
    template <int dim>
    class RefScalar : public Function<dim>
    {
    public:
      RefScalar()
        : Function<dim>(1)
      {}

      // Evaluate log-permeability at a single point
      virtual double
      value(const Point<dim>  &point,
            const unsigned int component) const override;

      // Evaluate log-permeability at multiple points
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double>           &values,
                 const unsigned int             component) const override;
    };

    template <int dim>
    double
    RefScalar<dim>::value(const Point<dim>  &point,
                          const unsigned int component) const
    {
      (void)component;

      const double factor     = 6.0;
      const double base_value = 0.5;
      const double phase      = std::sqrt(2.0) / 3.0;
      double       value;

      if (dim == 2)
        {
          // 2D: Smooth oscillatory pattern
          value = std::tanh((std::sin(factor * point[0] + phase) +
                             std::cos(2 * factor * point[1]) + sin(0) +
                             std::cos(phase) + sin(-phase)) *
                            (-1.0));
        }
      else if (dim == 3)
        {
          // 3D: Oscillatory pattern with rectangular inclusion
          value = std::tanh((std::sin(factor * point[0] + phase) +
                             std::cos(2 * factor * point[1]) +
                             sin(factor * point[2]) +
                             std::cos(1.5 * factor * point[2] + phase) +
                             sin(factor * point[2] - phase)) *
                            (point[1] * point[2] - 1.0));

          // Low-permeability inclusion region
          if (point[0] > -0.25 && point[0] < 0.25 && point[1] > -0.25 &&
              point[1] < 0.25 && point[2] > -0.9 && point[2] < -0.5)
            {
              value = base_value;
            }
        }

      return value;
    }

    template <int dim>
    void
    RefScalar<dim>::value_list(const std::vector<Point<dim>> &point,
                               std::vector<double>           &values,
                               const unsigned int             component) const
    {
      for (unsigned int i = 0; i < point.size(); ++i)
        {
          values[i] = this->value(point[i], component);
        }
    }

    // =========================================================================
    // get_k_mat: Compute permeability tensor K = exp(x) * I
    // Input:  rf_value = log-permeability x at a point
    // Output: K_mat = exp(x) * I (isotropic tensor)
    // =========================================================================
    template <int dim>
    void
    get_k_mat(double &rf_value, Tensor<2, dim> &K_mat)
    {
      K_mat = std::exp(rf_value) * unit_symmetric_tensor<dim>();
    }

    // =========================================================================
    // get_jacobi_inv_kmat: Derivative of K^{-1} w.r.t. log-permeability.
    //
    // Since K = exp(x) * I, we have K^{-1} = exp(-x) * I.
    // Therefore: d(K^{-1})/dx = -exp(-x) * I
    //
    // For the chain rule with FE basis: d(K^{-1})/dx_k = -exp(-x) * phi_k * I
    //
    // Input:  rf_value = log-permeability x at quadrature point
    //         grad_rf_x_value = shape function value phi_k(q)
    // Output: grad_inv_k_mat = -exp(-x) * phi_k * I
    // =========================================================================
    template <int dim>
    void
    get_jacobi_inv_kmat(double         &rf_value,
                        double          grad_rf_x_value,
                        Tensor<2, dim> &grad_inv_k_mat)
    {
      grad_inv_k_mat = -1.0 / std::exp(rf_value) *
                       unit_symmetric_tensor<dim>() * grad_rf_x_value;
    }

  } // namespace RandomMedium
} // namespace darcy

#endif // RANDOM_PERMEABILITY_H