# Section 5.3, wave steepening (the steepening figure of the paper).
#
#   julia --project=. -t 8 experiments/steepening/run.jl         # all three cases
#   julia --project=. -t 8 experiments/steepening/run.jl nl lin  # selected cases
#
# c = 1, α = 1e-6, zero initial data, Gaussian source at (0.15, 0.5) with
# r0 = 0.05, ω0 = 16π, T_s = 1; upwind flux, q_u = 4, q_v = 3, N = 64, T = 1.8, σ = 0.
# Cases:
#   nl     β = 8, absorbing boundary condition (weight √(1-2βv))
#   lin    β = 0, absorbing boundary condition
#   nl_dbc β = 8, homogeneous Dirichlet boundary condition
# Output per case in results/steepening/<case>/:
#   recv.csv    t and v_h at the receivers (0.45,0.5), (0.7,0.5), (0.85,0.5)
#   series.csv  t, Westervelt energy, max|v_h|, min(1 - 2βv_h)
#   grid.csv    x and y coordinates of the snapshot grid
#   snap_<t>.csv  v_h on that grid at t ≈ 0.5, 0.8, 1.1
# Cases run one after another; the nl case takes about 2.5 minutes with 8
# threads (-t 8) and about 11 minutes on one thread.
using WesterveltEDG, Printf, DelimitedFiles

const CASES = Dict("nl" => (8.0, :absorbing_2beta), "lin" => (0.0, :absorbing_2beta), "nl_dbc" => (8.0, :dirichlet))
const OUT = joinpath(@__DIR__, "..", "..", "results", "steepening")

function runcase(name)
    beta, boundary = CASES[name]
    t0 = time()
    H = solve(4, 64, 1.8; flux = :upwind, mixed = true, mode = :source, boundary = boundary,
              coefficients = constant_coefficients(c = 1.0, alpha = 1e-6, beta = beta),
              source = GaussianSource(x0 = (0.15, 0.5), r0 = 0.05, omega = 16π, Ts = 1.0),
              record_every = 32, receivers = [(0.45, 0.5), (0.7, 0.5), (0.85, 0.5)],
              snap_times = [0.5, 0.8, 1.1])
    d = joinpath(OUT, name)
    mkpath(d)
    writedlm(joinpath(d, "recv.csv"), hcat(H.recv_t, H.recv), ',')
    writedlm(joinpath(d, "series.csv"), hcat(H.t, H.energy, H.vmax, H.ndmin), ',')
    writedlm(joinpath(d, "grid.csv"), hcat(H.grid_x, H.grid_y), ',')
    for (ts, G) in H.snaps
        writedlm(joinpath(d, @sprintf("snap_%.2f.csv", ts)), G, ',')
    end
    @printf("%-7s beta=%.0f %-15s max|v|=%.4f min(1-2βv)=%.4f (%.0f s)\n",
            name, beta, boundary, maximum(H.vmax), minimum(H.ndmin), time() - t0)
end

for name in (isempty(ARGS) ? ["nl", "lin", "nl_dbc"] : ARGS)
    haskey(CASES, name) || error("unknown case $name (expected nl, lin or nl_dbc)")
    runcase(name)
end
