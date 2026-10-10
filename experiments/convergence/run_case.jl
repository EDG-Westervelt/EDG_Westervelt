# Section 5.1, convergence test: one (q_u, pairing, flux) case over N = 5..25.
#
#   julia --project=. experiments/convergence/run_case.jl <q_u> <mixed|matched> <upwind|central|alternating>
#
# Writes results/convergence/parts/q<q_u>_<pairing>_<flux>.csv. Use run_all.sh to
# run all 36 cases and merge them into results/convergence/results.csv.
#
# Manufactured solution u = cos(ωt) cos(2πx) sin(2πy), ω = 2√2π, coefficients of
# Section 5.1, T = 0.1, absorbing boundary condition with the weight √(1-2βv)
# (default) and σ = 0, RK4 with Δt = h/(10 q_u²).
using WesterveltEDG, Printf

const q  = parse(Int, ARGS[1])
const pairing = ARGS[2]
pairing in ("mixed", "matched") || error("pairing must be mixed or matched")
const fl = Symbol(ARGS[3])

const OUT = joinpath(@__DIR__, "..", "..", "results", "convergence", "parts")
mkpath(OUT)
const CSV = joinpath(OUT, "q$(q)_$(pairing)_$(fl).csv")
open(CSV, "w") do io; println(io, "q,pairing,flux,N,h,erru,errv,errux,errE,secs"); end

for N in 5:25
    t0 = time()
    e = try
        solve(q, N, 0.1; flux = fl, mixed = pairing == "mixed")
    catch err
        @warn "failed: q=$q $pairing $fl N=$N" err
        (erru = NaN, errv = NaN, errgradu = NaN, errE = NaN)
    end
    secs = time() - t0
    open(CSV, "a") do io
        @printf(io, "%d,%s,%s,%d,%.8e,%.10e,%.10e,%.10e,%.10e,%.1f\n",
                q, pairing, fl, N, 1.0/N, e.erru, e.errv, e.errgradu, e.errE, secs)
    end
    @printf("q=%d %-7s %-11s N=%-3d erru=%.3e errE=%.3e (%.0fs)\n", q, pairing, fl, N, e.erru, e.errE, secs)
    flush(stdout)
end
