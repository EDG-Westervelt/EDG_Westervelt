# PDE coefficients c(x,y), α(x,y), β(x,y) of the Westervelt equation.
# The first derivatives of c and α enter the manufactured source term only.
# Coefficients are evaluated during precomputation, never inside the RHS; on
# the faces from both sides (`_side_values!` in solver.jl), so that they may
# jump across mesh lines.

"""
    Coefficients(c, c_x, c_y, alpha, alpha_x, alpha_y, beta)

Sound speed `c`, diffusivity `alpha`, nonlinearity `beta` as functions
`(x, y) -> value`, together with the first derivatives of `c` and `alpha`
(needed for the manufactured source ``∇·(c²∇Φ)`` and ``∇·(α∇Φ)``).

The coefficients may be piecewise continuous with jumps on mesh lines
`x = k/N` or `y = j/N` (for example `(x, y) -> x < 0.5 ? 1.0 : 2.0` with an
even `N`). On a face, every element uses the limit of the coefficient from
its own side, obtained by evaluating the function at the face points shifted
by `2⁻⁴⁰ ≈ 9e-13` into the element; where these values agree with the
point value to the relative tolerance 1e-8 (plus 1e-8 times the largest
shifted value), the point value is used on both sides, so continuous
coefficients give the same numbers as before. The value a function returns
on the interface itself does not matter (`x < 0.5` and `x <= 0.5` give the
same result, and so does any other value there, also `Inf` or `NaN`). The derivatives `c_x`, `c_y`, `alpha_x`, `alpha_y` are those of the
pieces; they are only evaluated inside the elements. Jumps that cut through
elements are not detected: the scheme stays stable, but the accuracy drops
to that of a non-smooth solution.
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
