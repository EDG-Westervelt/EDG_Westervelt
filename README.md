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

The right-hand side is evaluated for all elements at once: the volume terms
`(c²∇φ, ∇u_h)`, `(α∇φ, ∇v_h)` are obtained by a few large matrix products
(sum factorization, without element matrices), every interior face is
visited once, and all work arrays are preallocated. The element mass
systems with the solution-dependent weight `1 - 2βv_h` are solved
matrix-free, by Chebyshev iteration on all elements at once (default) or by
conjugate gradients element by element (`mass_solver = :cg`). With Julia
started on several threads, each evaluation on a large enough mesh runs in
parallel on blocks of elements, with results identical to the serial run
(see *Multithreading*).

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
  `replace_coefficients`; they may jump across mesh lines (see
  *Coefficients*).
- `dt_fac`: the time step is `Δt ≈ h/(dt_fac q_u²)`; it does not scale with
  `c` or `α`, so raise `dt_fac` for faster or more viscous media (see
  *Coefficients*).
- `initial_profile`: in `:initial_value` mode, a NamedTuple of functions
  `(u0, u0_x, u0_y, u0_xx, u0_yy)` of `(x, y)` giving the initial displacement
  `u(0)` (default `Φ`).
- `mass_solver`: `:chebyshev` (default) or `:cg`, the solver of the element
  mass systems; both stop at the relative residual 1e-14.
- `nblocks`: number of element blocks evaluated in parallel (default
  `nothing`: chosen from the size of the problem, see *Multithreading*).

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

## Coefficients

`coefficients` is a `Coefficients(c, c_x, c_y, alpha, alpha_x, alpha_y, beta)`
of functions of `(x, y)`. `paper_coefficients()` gives those of Section 5.1,
and `constant_coefficients` and `replace_coefficients` build constant ones.
The derivatives of `c` and `α` are only used by the manufactured source of
`:mms` mode and by `∇·(c²∇u₀)` in `:initial_value` mode.

### Piecewise-continuous coefficients

`c`, `α` and `β` may jump across mesh lines `x = k/N` or `y = j/N`, as the
paper allows (`c`, `α` piecewise `W^{1,∞}` and `β` piecewise continuous on a
partition resolved by the mesh). For example, a two-layer medium with the
interface `x = 1/2` (a mesh line for even `N`):

```julia
cs(x, y) = x < 0.5 ? 1.0 : 0.5
z(x, y) = 0.0
co = Coefficients(cs, z, z, (x, y) -> 1e-6 * cs(x, y)^2, z, z, (x, y) -> x < 0.5 ? 0.0 : 4.0)
H = solve(3, 32, 1.0; mode = :source, coefficients = co, boundary = :wall)
```

- **Fluxes.** On an interior face, each element uses the coefficients on
  its own side in all face terms: the fluxes (2.9) of the paper with the
  one-sided values `c₁², c₂², α₁, α₂` in the averages, in the jump
  `[[c²∇u]] + [[α∇v]]` of `v*`, and in the terms `c²∇φ_u·n (v* - v)` and
  `α∇φ_v·n (v* - v)` of each element. The penalty weight is the mean
  `½(c₁² + c₂²) λ_u + ½(α₁ + α₂) λ_v`. The discrete energy identity (2.10)
  holds, so the scheme is conservative for central and alternating fluxes
  and dissipative for the others, as for continuous coefficients.
- **One-sided values.** The function is evaluated at the face quadrature
  points shifted by `δ = 2⁻⁴⁰ ≈ 9·10⁻¹³` into each element (on the physical
  boundary: into the domain). Where both shifted values agree with the value
  at the point itself to the relative tolerance `10⁻⁸` (plus `10⁻⁸` times
  the largest shifted value, for pieces where `α` or `β` is 0), the point
  value is used on both sides. Therefore:
  - continuous coefficients give the same numbers as before, bit for bit
    (the paper's runs and `data/` are unchanged);
  - the value a function returns on the interface itself does not matter:
    `x < 0.5` and `x <= 0.5` give the same result, and so does any other
    value there (also `Inf` or `NaN`);
  - an interface given as a decimal is found although the mesh line is
    computed in floating point (`x < 0.3` with `N = 10`, whose mesh line is
    `3·0.1 = 0.30000000000000004`);
  - a jump below about `10⁻⁸` relative counts as continuous; a smooth
    coefficient whose normal derivative exceeds about `10⁴` times its
    maximum takes the shifted values, with an error of the order of `δ|∇f|`.

  `pre.one_sided` (from `precompute`) tells whether some interior face has
  different values of `c²` or `α` on its two sides, or nonzero interface
  terms of `u₀` (see *Modes*); a jump of `β`
  alone does not count, since `β` enters only the volume and boundary terms.
  The right-hand side then uses the one-sided face kernel on all faces
  (measured: no slower than the kernel with one value per face point).
- **Modes.**
  - `:source`: no further condition.
  - `:initial_value`: the change of variables `ũ = u - u₀` contributes the
    interface terms `[[c²∇u₀]]` to the face fluxes, so `u₀` may have kinks on
    mesh lines and `c` may jump where `∂_n u₀ ≠ 0`. Inside the elements `u₀`
    must be smooth.
  - `:mms`: `c` and `α` must be continuous across the element faces (`β`
    may jump); `precompute` raises an error otherwise, since the smooth
    `Φ` does not satisfy the transmission conditions at a jump of `c` or
    `α`. Use `:initial_value` or `:source` for coefficients that jump.
- **Time step.** `Δt = h/(dt_fac q_u²)` does not scale with the
  coefficients; the default `dt_fac = 10` suits the paper's `c ≈ 1` and
  small `α`. For faster pieces, raise `dt_fac` roughly in proportion to
  `max c`, and further for large `α` (the term `∇·(α∇v)` limits `Δt` like
  `h²/α`, so more on finer meshes). For example, with `q_u = q_v = 2`,
  upwind fluxes and `N = 32`, `c = 2` needs `dt_fac = 20`, and so does
  `c = 1` with `α = 10⁻²`. The default is kept, so that the paper's runs are
  unchanged.
- **Not supported.** Jumps that cut through elements (off the mesh lines)
  are not detected. The scheme stays stable, since `c² > 0` at the
  quadrature points, but the accuracy drops to that of a non-smooth solution
  and depends on where the quadrature points fall. Interfaces must be mesh
  lines of the uniform Cartesian mesh.

## Changes since 0.3.0

- Piecewise-continuous `c`, `α`, `β` with jumps on mesh lines (one-sided face
  coefficients, see *Coefficients*). Before, both elements at a face used
  the single point value of `c²` and `α`; with a jump in `c` or `α` this was
  stable but in general not consistent (for the exact standing mode of a
  two-layer medium, the errors of the upwind and central fluxes did not
  decrease with `h`). Results for continuous coefficients are unchanged bit
  for bit.
- The default mass solver is now `:chebyshev` (it was `:cg`). Both solve
  to the relative residual 1e-14, and the results differ at that level: the
  errors of the regression tests, whose reference values in `data/` were
  computed with `:cg`, agree to about 1e-11 relative.
- Multithreaded evaluation with the keyword `nblocks` (see
  *Multithreading*).

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
| §5.2 α → 0 table | `julia --project=. -t 8 experiments/alpha_limit/run.jl` | `results/alpha_limit/order.csv` + printed table | about one minute with 8 threads (2–3 minutes on one) |
| §5.3 steepening figure | `julia --project=. -t 8 experiments/steepening/run.jl` | `results/steepening/<case>/` | about 2.5 minutes per case with 8 threads (about 11 minutes on one; measured for `nl`) |
| | `python3 experiments/steepening/plot.py` | `figures/steepening/*.pdf` | seconds |

All experiments use `Δt = h/(10 q_u²)`, as stated in the paper. The
`data/` directory holds the reference output used in the paper, all of it
produced with this step and the default boundary condition `:absorbing_2beta`
(the steepening comparison case `nl_dbc` uses `:dirichlet`) with v0.3.0
(2026-10-09; `data/alpha_limit` on 2026-10-12, with the compactly supported
initial data of `experiments/alpha_limit/run.jl`). The data of v0.2.0,
computed with the weight `√(1 - βv)` (and, for the α → 0 table, the initial
data `u(0) = cos(2πx) sin(2πy)`), are in the git history (commit `55b976b`);
the convergence rates are the same for both weights, and the steepening
receiver signals differ by about 1 %. The
plotting and table scripts read `results/` when it exists and `data/`
otherwise, so the paper's figures and tables can be rebuilt without rerunning
the simulations. A subset of the convergence sweep can be run with, for
example, `QS="1 2 3" bash experiments/convergence/run_all.sh`.

Run parameter sweeps such as the convergence cases as separate
single-threaded processes, as `run_all.sh` does, rather than as concurrent
tasks inside one Julia session; the multithreading below is meant for a
single long run.

## Multithreading

Start Julia with threads, for example `julia --project=. -t 8` (or set
`JULIA_NUM_THREADS=8`). `solve` then splits the elements into contiguous
blocks and evaluates every right-hand side in three parallel regions
(element blocks, blocks of mesh columns for the faces, element blocks
again), one task per block; the RK4 updates and, in `:source` mode, the
energy diagnostics run on the same blocks. The keyword `nblocks` sets the
number of blocks (at most `N²÷4`, so that a block has at least 4 elements),
and `nblocks = 1` is the serial code. The default `nblocks = nothing`
chooses from the work `N²(q_u+1)²`, because a parallel region costs a fixed
few tens of microseconds of synchronization: below 5000 the serial code,
below 20000 at most 4 blocks, above one block per Julia thread. For
`q_u = 4` this is serial up to 14×14 elements and 4 blocks up to 28×28; for
`q_u = 1` serial up to 35×35 and 4 blocks up to 70×70. The result does not
depend on `nblocks` or on the number of Julia threads: the runs are
identical bit for bit.
Every block does the same operations on its elements, and the Chebyshev
iteration takes the same number of steps on all elements. This relies on
the BLAS computing each entry of a matrix product the same way whatever the
size of the product, which the OpenBLAS shipped with Julia does (checked on
Apple silicon); with another BLAS the runs may differ in the last bits.

While more than one block is used, `solve`, `edg_rhs` and `rk4_step` set
BLAS to one thread (each task makes its own BLAS calls) and restore the
previous number when they return (after the last of several concurrent
runs). The number of BLAS
threads does not change the time stepping, but, as in earlier versions, it
can change the last bits of the inverse element stiffness that `precompute`
obtains from LAPACK; `solve` computes it with the session's number of BLAS
threads. When calling `edg_rhs!` or `rk4_step!` directly with a workspace
of more than one block (`ws.nb > 1`, which the default `Workspace(pre, N)`
gives on large meshes when Julia has several threads), call
`BLAS.set_num_threads(1)` (`using LinearAlgebra`) yourself.

On an Apple M4 Pro (10 performance cores) with 8 threads, one right-hand
side and one RK4 step take 6.3–6.8 times less time than with `nblocks = 1`,
both with BLAS at one thread, for `q_u = 2, 4, 6` on meshes of 100×100 and
150×150 elements, and about 6 times less on 64×64 for `q_u ≥ 3`. Against
the serial code with the session's default BLAS threads (10 here), which
makes the serial evaluation about 15% faster, the gain for `q_u = 4` on
150×150 is about 5.5 for a right-hand side and 5 for an RK4 step. The
u-equation reads one stored inverse per element (its upper triangle, the
inverse being symmetric) and is limited by memory bandwidth, so it gains
least. Small meshes gain less: the speedup is about
3–5.5 on 32×32 for `q_u ≥ 3`. Whole runs: `solve(4, 32, 0.1)` takes 1.2 s
instead of 3.4 s, and the `:source` run with `q_u = 4`, 64×64, `T = 0.05`
2.5 s instead of 13 s. For sweeps, separate processes remain the most
efficient. If processes and threads are combined, keep the number of
processes times the threads per process at most the number of physical
cores.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(julia_args = ["-t", "8"])'   # with 8 threads
```

The suite takes about 3–4 minutes. The tests check:

- the quadrature;
- that the zero initial state solves the initialization system for the
  manufactured solution;
- that the right-hand side agrees with the element-by-element implementation
  of v0.1.0 (`test/reference_rhs.jl`, dense LU mass solve, with the interior
  faces written for one-sided coefficients) for random states, all fluxes,
  both degree pairings, all modes and every boundary kind;
- regression against `data/convergence/results.csv` and, for `:absorbing`,
  against the data of v0.2.0;
- the boundary fluxes pointwise (consistency and the sign of the boundary
  energy term), that each side only affects its adjacent elements, the
  argument checks, and the discrete energy for walls and Dirichlet sides;
- the initialization and the convergence order on two small meshes for every
  boundary kind and a mixed configuration;
- both mass solvers against a dense solve for `1 - 2βv_h` down to 0.1, and
  whole runs with `mass_solver = :cg` and `:chebyshev`;
- that the right-hand side, RK4 steps and short runs are bitwise identical
  for every number of blocks (`nblocks`), for both mass solvers, all modes,
  every boundary kind and both degree pairings, that errors raised inside
  the parallel regions keep their type and message, that arrays of
  mismatched sizes are refused, and that the parallel energy diagnostics of
  `:source` mode equal the serial expressions bit for bit;
- that passing the default profile `Φ` as `initial_profile` reproduces the
  default `:initial_value` run;
- for piecewise coefficients: the one-sided face values (the point values,
  bit for bit, for continuous coefficients; the values of the pieces for
  different definitions of a jump, also with a huge, `Inf` or `NaN` value on
  another interface line; boundary values from inside), the reduction of the
  one-sided kernel to the plain one, the right-hand side against the
  reference with one-sided faces in all modes, the exact time derivatives of
  piecewise-quadratic states that satisfy the transmission conditions at a
  jump of `c` and `α` (no reference implementation involved), the
  semi-discrete energy identity (2.10) with one-sided coefficients to
  round-off for random states, the interface terms of `u₀` (the
  `:initial_value` right-hand side at `ũ = 0` equals the plain one at
  `u = u₀` for `u₀ ∈ U_h`), the convergence order for an exact two-layer
  standing mode, that `:mms` refuses jumps of `c` and `α`, the energy
  decay with absorbing sides, and the multithreading tests above with jumps.

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
                     (interior faces with one-sided coefficients)
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
