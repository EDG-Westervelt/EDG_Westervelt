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

The right-hand side is evaluated for all elements at once: the traces on the
element sides are obtained by a few large matrix products, every interior
face is visited once, and all work arrays are preallocated. The element mass
systems with the solution-dependent weight `1 - 2βv_h` are solved by
matrix-free conjugate gradients (see *Implementation notes* below).

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

See `?solve` for the full list. Lower-level, in-place building blocks are
`precompute`, `Workspace`, `edg_rhs!` and `rk4_step!` (allocating versions:
`edg_rhs`, `rk4_step`).

## Reproducing the paper

| Paper | Command | Output | Runtime |
|---|---|---|---|
| §5.1 convergence figure and table | `JOBS=8 bash experiments/convergence/run_all.sh` | `results/convergence/results.csv` | about 5 minutes with 8 processes (36 min on one core) |
| | `python3 experiments/convergence/make_table.py` | `figures/convergence/conv_table.tex` | seconds |
| | `python3 experiments/convergence/plot_panels.py` | `figures/convergence/*.pdf` | seconds |
| §5.2 α → 0 table | `julia --project=. experiments/alpha_limit/run.jl` | `results/alpha_limit/order.csv` + printed table | about a minute |
| §5.3 steepening figure | `julia --project=. experiments/steepening/run.jl` | `results/steepening/<case>/` | about 20 minutes per case |
| | `python3 experiments/steepening/plot.py` | `figures/steepening/*.pdf` | seconds |

All experiments use `Δt = h/(10 q_u²)`, as stated in the paper. The
`data/` directory holds the reference output used in the paper, all of it
produced with this step. `data/convergence/` was produced with the current
code (2026-10-07); the α → 0 and steepening data were produced with the
element-by-element version of the right-hand side (v0.1.0), which agrees with
the current one to rounding error. The
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
- that the right-hand side agrees with the element-by-element implementation
  of v0.1.0 (`test/reference_rhs.jl`, dense LU mass solve) for random states,
  all fluxes, both degree pairings, all modes and both boundary conditions;
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
test/                test suite; reference_rhs.jl is the v0.1.0 right-hand side
```

## Implementation notes

**Right-hand side.** `edg_rhs!` works in stages on all elements at once:
(1) element-last copies of the coefficients; (2) traces of `v_h`, `∂_n v_h`
and `∂_n u_h` on the four sides of every element, as products of the 1D
trace and evaluation matrices with the coefficients of all elements; (3) the
numerical fluxes on all faces by broadcasting, each interior face once with
its contributions to both neighbours; (4) the lift of the face terms back to
the elements; (5) volume terms and the u-equation element by element; the
manufactured source is a combination of four time-independent projections.
All arrays live in a `Workspace` allocated once per run, and RK4 is done in
place (`rk4_step!`).

**Mass systems.** The v-equation requires, on every element and in every
stage, the solution of `M(v_h) X = R` with
`M(v_h) = Φ' diag((1 - 2βv_q) W) Φ` (Φ: Legendre basis at the q_high-point
Gauss rule, W: weights, v_q: v_h at those points). The basis is orthonormal
and the rule is exact for the products, so `M = I - K` with
`K X = P' (G ∘ (P X P')) P`, `G = 2βv_q W`, and the eigenvalues of M lie in
`[min(1 - 2βv_q), max(1 - 2βv_q)]`: M is symmetric positive definite whenever
`1 - 2βv_h > 0` at the quadrature points.
The system is solved by conjugate gradients without assembling M, starting
from `X = R` (the solution for K = 0), to a relative residual of 1e-14. In
the convergence test `max|2βv| ≈ 1e-3` and CG needs about three iterations;
it stays below about 11 iterations on average even when `1 - 2βv_h = 0.1`.
A non-positive curvature `p'Mp ≤ 0`, i.e. an element mass matrix that is not
positive definite (which requires `1 - 2βv_h ≤ 0` somewhere in the element),
stops the run.

**Cost** (one core, N = 25, paper's convergence setting, per evaluation of
the right-hand side):

| q_u | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| v0.1.0 (element by element, LU) | 3.9 ms | 4.5 ms | 5.3 ms | 7.2 ms | 11.8 ms | 17.9 ms |
| current | 0.5 ms | 1.0 ms | 1.3 ms | 1.8 ms | 2.8 ms | 4.0 ms |

Memory allocated per evaluation dropped from 12–60 MB to about 0.1 MB. The
full q_u = 6, N = 25, T = 0.1 run takes 14.5 s instead of 62 s.

## License

MIT, see [LICENSE](LICENSE).
