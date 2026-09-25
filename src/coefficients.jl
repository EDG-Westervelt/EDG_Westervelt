# PDE coefficients c(x,y), α(x,y), β(x,y) of the Westervelt equation.
# The first derivatives of c and α enter the manufactured source term only.
# Coefficients are evaluated during precomputation, never inside the RHS.

"""
    Coefficients(c, c_x, c_y, alpha, alpha_x, alpha_y, beta)

Sound speed `c`, diffusivity `alpha`, nonlinearity `beta` as functions
`(x, y) -> value`, together with the first derivatives of `c` and `alpha`
(needed for the manufactured source ``∇·(c²∇Φ)`` and ``∇·(α∇Φ)``).
"""
struct Coefficients{C,CX,CY,A,AX,AY,B}
    c::C
    c_x::CX
    c_y::CY
    alpha::A
    alpha_x::AX
    alpha_y::AY
    beta::B
end

"""
    paper_coefficients()

The spatially varying coefficients of the convergence test (Section 5.1):

    c(x,y) = 1 + 0.1 cos(πx) cos(πy)
    α(x,y) = 1e-5 + 1e-6 cos(πx) cos(πy)
    β(x,y) = 1e-4 (1 + 0.1 sin(πx) sin(πy))
"""
paper_coefficients() = Coefficients(
    (x, y) -> 1 + 0.1 * cos(π * x) * cos(π * y),
    (x, y) -> -0.1π * sin(π * x) * cos(π * y),
    (x, y) -> -0.1π * cos(π * x) * sin(π * y),
    (x, y) -> 1e-5 + 1e-6 * cos(π * x) * cos(π * y),
    (x, y) -> -1e-6π * sin(π * x) * cos(π * y),
    (x, y) -> -1e-6π * cos(π * x) * sin(π * y),
    (x, y) -> 1e-4 * (1 + 0.1 * sin(π * x) * sin(π * y)),
)

_const(val) = (x, y) -> val
_zero(x, y) = zero(x)

"""
    replace_coefficients(coef; c=nothing, alpha=nothing, beta=nothing)

Return a copy of `coef` in which every coefficient given as a number is
replaced by that constant (and its derivatives by zero).
"""
function replace_coefficients(coef::Coefficients; c = nothing, alpha = nothing, beta = nothing)
    Coefficients(
        c === nothing ? coef.c   : _const(Float64(c)),
        c === nothing ? coef.c_x : _zero,
        c === nothing ? coef.c_y : _zero,
        alpha === nothing ? coef.alpha   : _const(Float64(alpha)),
        alpha === nothing ? coef.alpha_x : _zero,
        alpha === nothing ? coef.alpha_y : _zero,
        beta === nothing ? coef.beta : _const(Float64(beta)),
    )
end

"""
    constant_coefficients(; c, alpha, beta)

Constant coefficients.
"""
constant_coefficients(; c, alpha, beta) =
    replace_coefficients(paper_coefficients(); c = c, alpha = alpha, beta = beta)
