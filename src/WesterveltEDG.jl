"""
    WesterveltEDG

Energy-based discontinuous Galerkin (EDG) discretization of the Westervelt
equation with an absorbing boundary condition on the unit square.
"""
module WesterveltEDG

using LinearAlgebra
using SpecialFunctions: gamma

export solve, precompute, edg_rhs, rk4_step, difference_norms
export flux_parameters, GaussianSource
export Coefficients, paper_coefficients, replace_coefficients, constant_coefficients
export JacobiP, GradJacobiP, GaussQCofs

include("quadrature.jl")
include("coefficients.jl")
include("rhs.jl")
include("solver.jl")

end
