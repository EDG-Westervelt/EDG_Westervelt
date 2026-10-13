# Section 5.2, limiting behavior as α → 0 (the α-convergence table of the paper).
#
#   julia --project=. experiments/alpha_limit/run.jl
#
# Initial data u(0) = exp(1 - 1/(1 - |x - x0|²/R²)) for |x - x0| < R and 0
# otherwise, x0 = (0.5, 0.5), R = 0.25 (smooth and zero near ∂Ω, so the data are
# compatible with the boundary condition to any order), u_t(0) = 0, no source,
# homogeneous absorbing boundary condition with the weight √(1-2βv) (default,
# σ = 0); c and β as in Section 5.1, constant α.
# q_u = 6, q_v = 5, upwind flux, N = 20, T = 0.5 (the waves reach the boundary
# at t ≈ 0.25), Δt = h/(10 q_u²).
# The solutions for α ∈ {1e-5, ..., 1e-9} are compared with the numerical
# solution for α = 0 on the same mesh. Writes results/alpha_limit/order.csv.
# Runtime: about three minutes.
using WesterveltEDG, Printf

const X0 = (0.5, 0.5)
const R  = 0.25
# u(0) = g(s), s = |x - x0|²/R², with its first and second derivatives
s_(x, y) = ((x - X0[1])^2 + (y - X0[2])^2) / R^2
g(s)  = s < 1 ? exp(1 - 1 / (1 - s)) : 0.0
g1(s) = s < 1 ? -g(s) / (1 - s)^2 : 0.0
g2(s) = s < 1 ? g(s) * (1 / (1 - s)^4 - 2 / (1 - s)^3) : 0.0
const BUMP = (u0    = (x, y) -> g(s_(x, y)),
              u0_x  = (x, y) -> g1(s_(x, y)) * 2 * (x - X0[1]) / R^2,
              u0_y  = (x, y) -> g1(s_(x, y)) * 2 * (y - X0[2]) / R^2,
              u0_xx = (x, y) -> g2(s_(x, y)) * (2 * (x - X0[1]) / R^2)^2 + g1(s_(x, y)) * 2 / R^2,
              u0_yy = (x, y) -> g2(s_(x, y)) * (2 * (y - X0[2]) / R^2)^2 + g1(s_(x, y)) * 2 / R^2)

const ALPHAS = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
run(α) = solve(6, 20, 0.5; flux = :upwind, mixed = true, mode = :initial_value,
               coefficients = replace_coefficients(paper_coefficients(); alpha = α),
               initial_profile = BUMP, return_state = true)

ref = run(0.0)
println("solved alpha = 0")
res = map(ALPHAS) do α
    d = difference_norms(run(α), ref)
    @printf("solved alpha = %.0e\n", α)
    d
end

const OUT = joinpath(@__DIR__, "..", "..", "results", "alpha_limit")
mkpath(OUT)
open(joinpath(OUT, "order.csv"), "w") do io
    println(io, "alpha,erru,errv,errE")
    for (α, d) in zip(ALPHAS, res)
        @printf(io, "%.6e,%.10e,%.10e,%.10e\n", α, d.erru, d.errv, d.errE)
    end
end

println("\n alpha    ‖e_u‖_L2    order   ‖e‖_en     order")
for i in eachindex(ALPHAS)
    ou = i == 1 ? "  -- " : @sprintf("%5.2f", log10(res[i-1].erru / res[i].erru))
    oE = i == 1 ? "  -- " : @sprintf("%5.2f", log10(res[i-1].errE / res[i].errE))
    @printf(" %.0e  %.2e   %s   %.2e   %s\n", ALPHAS[i], res[i].erru, ou, res[i].errE, oE)
end
