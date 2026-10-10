# Section 5.2, limiting behavior as α → 0 (the α-convergence table of the paper).
#
#   julia --project=. experiments/alpha_limit/run.jl
#
# Initial data u(0) = cos(2πx) sin(2πy), u_t(0) = 0, no source, homogeneous
# absorbing boundary condition with the weight √(1-2βv) (default, σ = 0); c and
# β as in Section 5.1, constant α.
# q_u = 6, q_v = 5, upwind flux, N = 20, T = 0.1, Δt = h/(10 q_u²).
# The solutions for α ∈ {1e-5, ..., 1e-9} are compared with the numerical
# solution for α = 0 on the same mesh. Writes results/alpha_limit/order.csv.
# Runtime: about a minute.
using WesterveltEDG, Printf

const ALPHAS = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
run(α) = solve(6, 20, 0.1; flux = :upwind, mixed = true, mode = :initial_value,
               coefficients = replace_coefficients(paper_coefficients(); alpha = α),
               return_state = true)

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

println("\n alpha    ‖e_u‖_L2    order   ‖e‖_E      order")
for i in eachindex(ALPHAS)
    ou = i == 1 ? "  -- " : @sprintf("%5.2f", log10(res[i-1].erru / res[i].erru))
    oE = i == 1 ? "  -- " : @sprintf("%5.2f", log10(res[i-1].errE / res[i].errE))
    @printf(" %.0e  %.2e   %s   %.2e   %s\n", ALPHAS[i], res[i].erru, ou, res[i].errE, oE)
end
