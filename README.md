# WesterveltEDG.jl

Julia code for the energy-based discontinuous Galerkin (EDG) method for the
Westervelt equation with absorbing boundary conditions. It reproduces the
numerical experiments in Section 5 of

> Y. Wang, L. Zhang, *Energy-based discontinuous Galerkin methods for the
> Westervelt equation with absorbing boundary conditions*, in preparation.

The solver works on the unit square with a uniform Cartesian mesh. It uses
tensor-product Legendre bases of degree `q_u` for `u_h` and `q_v ∈ {q_u - 1, q_u}`
for `v_h`, and integrates in time with classical RK4. Upwind, central and
alternating interelement fluxes are available. Each side of the square takes
its own boundary condition: the absorbing boundary condition of the paper
with the weight `√(1 - 2βv)` (default), the same condition with the
alternative weight `√(1 - βv)` (Remark 4.3 of the paper), a Dirichlet
condition, or a reflecting wall (see *Boundary conditions* below).

The right-hand side is evaluated for all elements at once: the traces on the
element sides are obtained by a few large matrix products, every interior
face is visited once, and all work arrays are preallocated. The element mass
systems with the solution-dependent weight `1 - 2βv_h` are solved by
matrix-free conjugate gradients.

## Installation

The package requires Julia ≥ 1.10. To reproduce the paper, clone the
repository and instantiate the environment:

```bash
git clone https://github.com/EDG-Westervelt/EDG_Westervelt.git
cd EDG_Westervelt
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The examples below assume Julia started with `julia --project=.` in the
repository root. To use the solver from another environment instead:

```julia
using Pkg
Pkg.add(url = "https://github.com/EDG-Westervelt/EDG_Westervelt")
```

The figure scripts need Python 3 with `numpy` and `matplotlib`
(`pip install -r requirements.txt`). If a LaTeX installation is found,
labels are typeset with LaTeX, as in the paper.

## Quick start

```julia
using WesterveltEDG

# Manufactured-solution test of Section 5.1: q_u = 3, q_v = 2, upwind flux, N = 10, T = 0.1
e = solve(3, 10, 0.1; flux = :upwind, mixed = true)
e.erru, e.errv, e.errgradu, e.errE      # ‖e_u‖, ‖e_v‖, ‖∇_h e_u‖, ‖e‖_en at t = T
# (1.0718e-4, 4.5773e-3, 7.1300e-3, 5.9884e-3): the row q = 3, mixed, upwind,
# N = 10 of data/convergence/results.csv
```

The main keywords of `solve` are:

- `mode`: `:mms`, `:initial_value` or `:source`.
- `flux`: `:upwind`, `:central` or `:alternating`, or custom parameters
  `flux_params = (theta, tau, lambda_u, lambda_v)`.
- `boundary`: one boundary kind for all four sides, or a NamedTuple
  `(bot = …, top = …, lft = …, rgt = …)` for the sides y = 0, y = 1, x = 0,
  x = 1; the kinds are `:absorbing_2beta` (default), `:absorbing`,
  `:dirichlet` and `:wall`.
- `sigma`, `sigma_dirichlet`, `kappa`: parameters of the absorbing,
  Dirichlet and wall fluxes.
- `coefficients`: see `paper_coefficients`, `constant_coefficients` and
  `replace_coefficients`.
- `dt_fac`: the time step is `Δt ≈ h/(dt_fac q_u²)`.

For example, three reflecting walls and an absorbing bottom side, with a
Gaussian source:

```julia
H = solve(4, 32, 1.0; mode = :source, kappa = 0.5,
          boundary = (bot = :absorbing_2beta, top = :wall, lft = :wall, rgt = :wall),
          coefficients = constant_coefficients(c = 1.0, alpha = 1e-6, beta = 8.0))
```

See `?solve` for the full list. Lower-level, in-place building blocks are
`precompute`, `Workspace`, `edg_rhs!` and `rk4_step!` (allocating versions:
`edg_rhs`, `rk4_step`).

## Boundary conditions

With `ψ_h = (c²∇u_h + α∇v_h)·n` and the boundary data of the chosen mode,
every kind defines the total numerical normal flux `((c²∇u)* + (α∇v)*)·n`
and the trace `v*` on its side. Each side is treated separately, so any
combination of kinds is allowed.

| Kind | Condition | Flux | Parameter |
|---|---|---|---|
| `:absorbing_2beta` | `c √(1 - 2βv) v + ψ = 0` | eq. (2.12) of the paper | `-1 ≤ sigma ≤ 1` |
| `:absorbing` | `c √(1 - βv) v + ψ = 0` | eq. (2.12) with `√(1 - βv)` | `-1 ≤ sigma ≤ 1` |
| `:dirichlet` | `u = g` | `v* = √(1 - βg_t) g_t / √(1 - βv_h)`, penalty `sigma_dirichlet` | `sigma_dirichlet ≥ 0` |
| `:wall` | `ψ = g₂` | total flux `g₂`, `v* = v_h - (κ/c)(ψ_h - g₂)` | `kappa ≥ 0` |

For homogeneous data, the boundary contribution to the discrete energy is
non-positive in every case; it vanishes for walls with `kappa = 0` and
Dirichlet sides with `sigma_dirichlet = 0`. Values outside the stated
ranges raise an error, as does `1 - 2βv ≤ 0` (`1 - βv ≤ 0` for `:absorbing`
and `:dirichlet`) for the trace `v_h` or the boundary data at a face
quadrature point of an absorbing or Dirichlet side. `sigma_dirichlet` defaults to `sigma`
and `kappa` to 0. The boundary data are those of the manufactured solution
in `:mms` mode (the keyword `phase` shifts it so that the data are non-zero
on every side), `u = u(0)` on Dirichlet sides and zero data otherwise in
`:initial_value` mode, and zero data in `:source` mode. The error analysis
of the paper covers the two absorbing kinds (`:absorbing` through Remark 4.3);
the convergence theorem for the nonlinear scheme (Theorem 4.2) assumes
`-1 < sigma ≤ 1`.

## Changes in 0.3.0

- Boundary conditions per side: `:absorbing_2beta`, `:absorbing`,
  `:dirichlet`, `:wall`, with the keywords `sigma_dirichlet`, `kappa` and
  `phase`.
- The default boundary condition is now `:absorbing_2beta` (weight
  `√(1 - 2βv)`, eq. (1.1b) of the paper); v0.1 and v0.2 used `:absorbing`
  (weight `√(1 - βv)`). `data/` was regenerated accordingly.
- Arguments outside the stable ranges (`|sigma| > 1` on absorbing sides,
  `sigma_dirichlet < 0`, `kappa < 0`) and a non-positive boundary weight at a
  face point now raise an error.

## Reproducing the paper

| Paper | Command | Output | Runtime |
|---|---|---|---|
| §5.1 convergence figure and table | `JOBS=8 bash experiments/convergence/run_all.sh` | `results/convergence/results.csv` | about 5 minutes with 8 processes (36 min on one core) |
| | `python3 experiments/convergence/make_table.py` | `figures/convergence/conv_table.tex` | seconds |
| | `python3 experiments/convergence/plot_panels.py` | `figures/convergence/*.pdf` | seconds |
| §5.2 α → 0 table | `julia --project=. experiments/alpha_limit/run.jl` | `results/alpha_limit/order.csv` + printed table | about a minute |
| §5.3 steepening figure | `julia --project=. experiments/steepening/run.jl` | `results/steepening/<case>/` | about 20–25 minutes per case |
| | `python3 experiments/steepening/plot.py` | `figures/steepening/*.pdf` | seconds |

All experiments use `Δt = h/(10 q_u²)`, as stated in the paper. The
`data/` directory holds the reference output used in the paper, all of it
produced with this step and the default boundary condition `:absorbing_2beta`
(the steepening comparison case `nl_dbc` uses `:dirichlet`) with v0.3.0
(2026-10-09). The data of v0.2.0, computed with the weight `√(1 - βv)`, are in
the git history (commit `55b976b`); the convergence rates and the α → 0 table are the same for
both weights, and the steepening receiver signals differ by about 1 %. The
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

The suite takes about 2–3 minutes. The tests check:

- the quadrature;
- that the zero initial state solves the initialization system for the
  manufactured solution;
- that the right-hand side agrees with the element-by-element implementation
  of v0.1.0 (`test/reference_rhs.jl`, dense LU mass solve) for random states,
  all fluxes, both degree pairings, all modes and every boundary kind;
- regression against `data/convergence/results.csv` and, for `:absorbing`,
  against the data of v0.2.0;
- the boundary fluxes pointwise (consistency and the sign of the boundary
  energy term), that each side only affects its adjacent elements, the
  argument checks, and the discrete energy for walls and Dirichlet sides;
- the initialization and the convergence order on two small meshes for every
  boundary kind and a mixed configuration.

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
results/, figures/   written by the drivers and figure scripts (not tracked)
```

## Citation

If you use this code, please cite the paper:

```bibtex
@unpublished{WangZhang-EDG-Westervelt,
  author = {Yiwei Wang and Lu Zhang},
  title  = {Energy-based discontinuous {G}alerkin methods for the {W}estervelt
            equation with absorbing boundary conditions},
  note   = {In preparation. Code: \url{https://github.com/EDG-Westervelt/EDG_Westervelt}}
}
```

## License

MIT, see [LICENSE](LICENSE).
