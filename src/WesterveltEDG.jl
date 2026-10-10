"""
    WesterveltEDG

Energy-based discontinuous Galerkin (EDG) discretization of the Westervelt
equation on the unit square with absorbing, Dirichlet or reflecting-wall
boundary conditions, chosen per side.
"""
module WesterveltEDG

using LinearAlgebra
using SpecialFunctions: gamma

export solve, precompute, edg_rhs, edg_rhs!, rk4_step, rk4_step!, Workspace, difference_norms
export flux_parameters, GaussianSource
export Coefficients, paper_coefficients, replace_coefficients, constant_coefficients
export JacobiP, GradJacobiP, GaussQCofs

include("quadrature.jl")
include("coefficients.jl")
include("rhs.jl")
include("solver.jl")

end
