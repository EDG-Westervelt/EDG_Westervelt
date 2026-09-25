# WesterveltEDG.jl

Julia code for the energy-based discontinuous Galerkin (EDG) method for the
Westervelt equation with absorbing boundary conditions. It reproduces the
numerical experiments in Section 5 of

> Y. Wang, L. Zhang, *Energy-based discontinuous Galerkin methods for the
> Westervelt equation with absorbing boundary conditions*, in preparation.

The solver works on the unit square with a uniform Cartesian mesh. It uses
tensor-product Legendre bases of degree `q_u` for `u_h` and `q_v ∈ {q_u - 1, q_u}`
for `v_h`, and integrates in time with classical RK4. Upwind, central and
alternating interelement fluxes are available. On the physical boundary the
solver imposes the absorbing boundary condition with weight `√(1 - βv)`, or
optionally a homogeneous Dirichlet condition.

## Installation

The package requires Julia ≥ 1.10. From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The figure scripts need Python 3 with `numpy` and `matplotlib`
(`pip install -r requirements.txt`). If a LaTeX installation is found,
labels are typeset with LaTeX, as in the paper.

## Quick start

```julia
using WesterveltEDG

# Manufactured-solution test of Section 5.1: q_u = 3, q_v = 2, upwind flux, N = 10, T = 0.1
e = solve(3, 10, 0.1; flux = :upwind, mixed = true)
e.erru, e.errv, e.errgradu, e.errE      # ‖e_u‖, ‖e_v‖, ‖∇_h e_u‖, ‖e‖_E at t = T
```

The main keywords of `solve` are:

- `mode`: `:mms`, `:initial_value` or `:source`.
- `flux`: `:upwind`, `:central` or `:alternating`, or custom parameters
  `flux_params = (theta, tau, lambda_u, lambda_v)`.
- `boundary`: `:absorbing` or `:dirichlet`.
- `sigma`: the boundary-flux parameter σ.
- `coefficients`: see `paper_coefficients`, `constant_coefficients` and
  `replace_coefficients`.
- `dt_fac`: the time step is `Δt ≈ h/(dt_fac q_u²)`.

See `?solve` for the full list.

## Reproducing the paper

| Paper | Command | Output | Runtime |
|---|---|---|---|
| §5.1 convergence figure and table | `JOBS=4 bash experiments/convergence/run_all.sh` | `results/convergence/results.csv` | a few hours (q_u = 6 dominates) |
| | `python3 experiments/convergence/make_table.py` | `figures/convergence/conv_table.tex` | seconds |
| | `python3 experiments/convergence/plot_panels.py` | `figures/convergence/*.pdf` | seconds |
| §5.2 α → 0 table | `julia --project=. experiments/alpha_limit/run.jl` | `results/alpha_limit/order.csv` + printed table | minutes |
| §5.3 steepening figure | `julia --project=. experiments/steepening/run.jl` | `results/steepening/<case>/` | about an hour per case |
| | `python3 experiments/steepening/plot.py` | `figures/steepening/*.pdf` | seconds |

All experiments use `Δt = h/(10 q_u²)`, as stated in the paper. The
`data/` directory holds the reference output used in the paper. Some of
those runs (the upwind cases with `q_u ≥ 5` in `data/convergence/` and the
α → 0 runs in `data/alpha_limit/`) used the smaller step `Δt = h/(15 q_u²)`,
so rerunning these cases is expected to agree closely with the reference
values, but not bit for bit. The
plotting and table scripts read `results/` when it exists and `data/`
otherwise, so the paper's figures and tables can be rebuilt without rerunning
the simulations. A subset of the convergence sweep can be run with, for
example, `QS="1 2 3" bash experiments/convergence/run_all.sh`.

Run the convergence cases as separate processes, as `run_all.sh` does, rather
than as concurrent `Threads.@spawn` tasks inside one Julia session.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests check:

- the quadrature;
- that the zero initial state solves the initialization system for the
  manufactured solution;
- regression against `data/convergence/results.csv`;
- the convergence order on two small meshes.

## Layout

```
src/
  WesterveltEDG.jl   module and exports
  quadrature.jl      Legendre/Jacobi polynomials, Gauss–Legendre rules
  coefficients.jl    c, α, β and the Section 5.1 coefficients
  rhs.jl             semi-discrete EDG right-hand side
  solver.jl          precomputation, RK4, errors, source-mode diagnostics
experiments/         drivers and figure scripts for Sections 5.1–5.3
data/                reference results used in the paper
test/                test suite
```

## License

MIT, see [LICENSE](LICENSE).
