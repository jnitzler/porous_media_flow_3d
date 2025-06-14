// random_permeability.h
#ifndef RANDOM_PERMEABILITY_H
#define RANDOM_PERMEABILITY_H

#include "darcy.h"

// include the random field implementation
// #include "../random_fields/rbf_fourier_expansion.h"
// #include "../random_fields/rbf_kle.h"

namespace darcy {
namespace RandomMedium {

// create a reference permeability field
template <int dim>
class RefScalar : public Function<dim> {
 public:
  RefScalar() : Function<dim>(1) {}
  virtual double value(const Point<dim> &point,
                       const unsigned int component) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component) const override;
};

template <int dim>
double RefScalar<dim>::value(const Point<dim> &point,
                             const unsigned int component) const {
  (void)component;
  // get the random field values
  double factor = 4.0;
  double value;
  value = 1.1 + std::sin(factor * point[0]) * std::cos(2 * factor * point[1]) *
                    sin(factor * point[2]);
  return value;
}

template <int dim>
void RefScalar<dim>::value_list(const std::vector<Point<dim>> &point,
                                std::vector<double> &values,
                                const unsigned int component) const {
  (void)component;
  for (unsigned int i = 0; i < point.size(); ++i) {
    this->value(point[i], values[i]);
  }
}

template <int dim>
void get_k_mat(double &rf_value, Tensor<2, dim> &K_mat) {
  K_mat = std::exp(rf_value) * unit_symmetric_tensor<dim>();
}

// ---------- derivatives -----------------------------------
template <int dim>
void get_jacobi_inv_kmat(double &rf_value, double grad_rf_x_value,
                         Tensor<2, dim> &grad_inv_k_mat) {
  grad_inv_k_mat = -1 / (std::exp(rf_value)) * unit_symmetric_tensor<dim>() *
                   grad_rf_x_value;
}  // end function get jacobi inv kmat

}  // end namespace RandomMedium
}  // end namespace darcy
#endif