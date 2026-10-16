# Precomputation, RK4 time stepping, error evaluation and diagnostics for the
# EDG discretization of the Westervelt equation on the unit square.

"""
    flux_parameters(flux::Symbol)

Interelement flux parameters `(theta, tau, lambda_u, lambda_v)` of the three
flux families used in the paper:

| `flux`         | θ   | τ | λ_u | λ_v |
|:---------------|:----|:--|:----|:----|
| `:upwind`      | 1   | 1 | 1   | 1   |
| `:alternating` | 1   | 0 | 0   | 0   |
| `:central`     | 1/2 | 0 | 0   | 0   |
"""
function flux_parameters(flux::Symbol)
    flux == :upwind      && return (theta = 1.0, tau = 1.0, lambda_u = 1.0, lambda_v = 1.0)
    flux == :alternating && return (theta = 1.0, tau = 0.0, lambda_u = 0.0, lambda_v = 0.0)
    flux == :central     && return (theta = 0.5, tau = 0.0, lambda_u = 0.0, lambda_v = 0.0)
    error("unknown flux $flux (expected :upwind, :alternating or :central)")
end

# Boundary kinds, chosen per side. The first three are members of the family
#     ρ = η (c²∇u + α∇v)·n + γ c √(1-kβv) v,   η = √(1-γ²),
#     ((c²∇u)* + (α∇v)*)·n = (c²∇u_h + α∇v_h)·n - (η + σγ) ρ,
#     v* = v_h - (γ - ση) ρ / (c √(1-kβv_h)),
# with ρ the residual against the boundary data:
# - :absorbing_2beta  γ = 1/√2, k = 2: absorbing condition with the weight
#                     √(1-2βv) of the paper's (1.1b), -1 ≤ σ ≤ 1 (default);
# - :absorbing        γ = 1/√2, k = 1: the alternative weight √(1-βv) of the
#                     paper's Remark 4.3 (the default of v0.1 and v0.2);
# - :dirichlet        γ = 1 (η = 0), k = 1: v* = √(1-βg) g / √(1-βv_h) for the
#                     boundary data g of v (so v* = 0 for homogeneous data),
#                     with the penalty σ_D = sigma_dirichlet ≥ 0;
# - :wall             reflecting wall (c²∇u + α∇v)·n = g₂ (see `_wall_face`),
#                     dissipation κ = kappa ≥ 0.
const BOUNDARY_KINDS = (:absorbing_2beta, :absorbing, :dirichlet, :wall)

function _boundary_side(kind::Symbol, sigma::Float64, sigma_dirichlet::Float64, kappa::Float64)
    kind === :wall && return (kind = kind, gamma_abc = 0.0, eta_abc = 1.0, sigma = 0.0, kb = 1.0, kappa = kappa)
    gamma_abc = kind === :dirichlet ? 1.0 : 1.0 / sqrt(2.0)
    return (kind = kind, gamma_abc = gamma_abc, eta_abc = sqrt(1.0 - gamma_abc^2),
            sigma = kind === :dirichlet ? sigma_dirichlet : sigma,
            kb = kind === :absorbing_2beta ? 2.0 : 1.0, kappa = kappa)
end

# Per-side parameters (bot, top, lft, rgt) from `boundary`: a kind for all four
# sides, or a NamedTuple with the keys bot (y = 0), top (y = 1), lft (x = 0) and
# rgt (x = 1) in any order.
function _boundary_sides(boundary, sigma::Float64, sigma_dirichlet::Float64, kappa::Float64)
    names = (:bot, :top, :lft, :rgt)
    if boundary isa Symbol
        kinds = (boundary, boundary, boundary, boundary)
    else
        (length(boundary) == 4 && issetequal(keys(boundary), names)) ||
            error("boundary must be a kind or a NamedTuple with the keys bot, top, lft, rgt")
        kinds = map(s -> boundary[s], names)
    end
    for kind in kinds
        kind in BOUNDARY_KINDS || error("unknown boundary $kind (expected one of $(BOUNDARY_KINDS))")
    end
    any(k -> k in (:absorbing, :absorbing_2beta), kinds) && !(-1.0 <= sigma <= 1.0) &&
        error("absorbing sides need -1 ≤ sigma ≤ 1 (got $sigma)")
    :dirichlet in kinds && !(sigma_dirichlet >= 0.0) &&
        error("Dirichlet sides need sigma_dirichlet ≥ 0 (got $sigma_dirichlet; it defaults to sigma)")
    kappa >= 0.0 || error("kappa must be ≥ 0 (got $kappa)")
    return NamedTuple{names}(map(k -> _boundary_side(k, sigma, sigma_dirichlet, kappa), kinds))
end

"""
    GaussianSource(; x0 = (0.15, 0.5), r0 = 0.05, amplitude = 1.0, omega = 16π, Ts = 1.0)

Localized, temporally modulated source used in the wave-steepening test
(Section 5.3):

    f(t, x) = amplitude · exp(-|x - x0|² / r0²) · sin(ω t) · sin²(π t / Ts),  0 ≤ t ≤ Ts,

and `f = 0` for `t > Ts`.
"""
Base.@kwdef struct GaussianSource
    x0::Tuple{Float64,Float64} = (0.15, 0.5)
    r0::Float64 = 0.05
    amplitude::Float64 = 1.0
    omega::Float64 = 16π
    Ts::Float64 = 1.0
end

"""
    solve(q_u, N, T; kwargs...)

Integrate the semi-discrete EDG scheme on `(0,1)²` with an `N×N` Cartesian
mesh (`h = 1/N`) up to time `T`, using tensor-product Legendre polynomials of
degree `q_u` for `u_h` and `q_v = q_u - 1` (`mixed = true`) or `q_v = q_u`
(`mixed = false`) for `v_h`. Time integration is the classical RK4 with
`Δt ≈ h / (dt_fac q_u²)`.

# Modes
- `mode = :mms` (default): manufactured solution `u = cos(ωt) Φ(x)` with
  `Φ = cos(2π kx x) sin(2π ky y)`, `ω = 2π √(kx² + ky²)`; the source and the
  boundary data are chosen consistently (Section 5.1). Returns
  `(erru, errv, errgradu, errE)`, the L² errors of `u`, `v`, the broken
  H¹-seminorm error of `u` and the energy error
  `‖e‖_en² = ½‖c∇_h e_u‖² + ½‖e_v‖²` at time `T`.
- `mode = :initial_value`: initial data `u(0) = Φ`, `u_t(0) = 0`, no source,
  homogeneous boundary conditions except `u = Φ` on Dirichlet sides
  (Section 5.2).
- `mode = :source`: zero initial data, homogeneous boundary conditions and the
  source `source::GaussianSource` (Section 5.3). Returns time series (see
  below).

# Keywords
- `flux = :upwind`, `flux_params = nothing`: interelement flux; a NamedTuple
  `(theta, tau, lambda_u, lambda_v)` overrides [`flux_parameters`](@ref).
- `boundary = :absorbing_2beta`: the boundary kind of all four sides, or a
  NamedTuple `(bot, top, lft, rgt)` of kinds for the sides y = 0, y = 1,
  x = 0, x = 1. Kinds: `:absorbing_2beta` (weight √(1-2βv)), `:absorbing`
  (weight √(1-βv)), `:dirichlet`, `:wall` (see *Boundary conditions* in the
  README).
- `sigma = 0.0`: flux parameter of absorbing sides, -1 ≤ σ ≤ 1.
- `sigma_dirichlet = sigma`: penalty of Dirichlet sides, ≥ 0.
- `kappa = 0.0`: dissipation of wall sides, ≥ 0.
- `phase = (0.0, 0.0)`: shifts `Φ` to `cos(2π kx x + px) sin(2π ky y + py)`.
- `initial_profile = nothing`: in `:initial_value` mode, a NamedTuple of
  functions `(u0, u0_x, u0_y, u0_xx, u0_yy)` of `(x, y)` that replaces `Φ` as
  the initial displacement `u(0)` (with `u_t(0) = 0`).
- `coefficients = paper_coefficients()`: see [`Coefficients`](@ref).
- `kx = 1, ky = 1`: wave numbers of Φ.
- `dt_fac = 10`.
- `mass_solver = :chebyshev`: solver of the element mass systems,
  `:chebyshev` (Chebyshev iteration on all elements at once) or `:cg`
  (conjugate gradients, element by element); both to the relative residual
  1e-14.
- `nblocks = nothing`: number of element blocks evaluated as parallel tasks
  (at most `N²÷4`); the default chooses from the size of the problem, one
  block per Julia thread on large meshes (see [`Workspace`](@ref)). The
  results do not depend on it. Start Julia with threads (`julia -t 8`) for a
  speedup. While more than one block is used, BLAS runs with one thread; the
  previous number of BLAS threads is restored when `solve` returns.
- `return_state = false`: in `:mms`/`:initial_value` mode, return the final
  coefficient arrays and quadrature operators instead of the errors
  (see [`difference_norms`](@ref)).
- `source`, `receivers`, `snap_times`, `slice_y`, `record_every`: `:source` mode
  output controls.

The displacement is integrated as `ũ = u - Φ` with `ũ(0) = 0`, `v(0) = 0`; for
the manufactured solution this zero state solves the nonlinear
initialization system of the paper exactly (see `test/runtests.jl`).
"""
function solve(q::Int, N::Int, T::Float64;
               mixed::Bool = true,
               flux::Symbol = :upwind,
               flux_params = nothing,
               sigma::Float64 = 0.0,
               boundary::Union{Symbol,NamedTuple} = :absorbing_2beta,
               sigma_dirichlet::Float64 = sigma,
               kappa::Float64 = 0.0,
               phase::Tuple{Float64,Float64} = (0.0, 0.0),
               coefficients::Coefficients = paper_coefficients(),
               mode::Symbol = :mms,
               kx::Int = 1, ky::Int = 1,
               dt_fac::Real = 10,
               mass_solver::Symbol = :chebyshev,
               nblocks::Union{Nothing,Integer} = nothing,
               initial_profile = nothing,
               return_state::Bool = false,
               source::GaussianSource = GaussianSource(),
               record_every::Int = 20,
               slice_y::Float64 = 0.5,
               receivers = [(0.45, 0.5), (0.7, 0.5), (0.85, 0.5)],
               snap_times = Float64[])

    mode in (:mms, :initial_value, :source) ||
        error("unknown mode $mode (expected :mms, :initial_value or :source)")
    pre, geo = _with_session_blas() do
        precompute(q, N; mixed = mixed, flux = flux, flux_params = flux_params,
                   sigma = sigma, boundary = boundary, sigma_dirichlet = sigma_dirichlet,
                   kappa = kappa, phase = phase, coefficients = coefficients,
                   mode = mode, kx = kx, ky = ky, source = source,
                   mass_solver = mass_solver, initial_profile = initial_profile)
    end

    h  = 1.0 / N
    u  = zeros(N, N, pre.qu1, pre.qu1)     # ũ = u - u₀
    v  = zeros(N, N, pre.qv1, pre.qv1)
    dt = h / (dt_fac * q^2)
    nsteps = round(Int, T / dt)
    dt = T / nsteps

    ws = Workspace(pre, N; nblocks = nblocks)
    counted = _blas_single!(ws)
    try
        mode == :source && return _run_source(u, v, dt, nsteps, N, h, pre, geo, ws;
                                              record_every = record_every, slice_y = slice_y,
                                              receivers = receivers, snap_times = snap_times)
        t = 0.0
        for _ in 1:nsteps
            rk4_step!(u, v, t, dt, N, h, pre, ws)
            t += dt
        end
    finally
        _blas_restore(counted)
    end

    if return_state
        return (u = u, v = v, N = N, h = h,
                P_u = pre.P_u, P_v = pre.P_v, S_u = geo.S_u, W2d = geo.W2d, err_c2 = geo.err_c2)
    end
    mode == :mms || error("errors are only available in :mms mode; use return_state = true")
    return _mms_errors(u, v, T, N, h, pre, geo, kx, ky, phase)
end

# One BLAS thread while a workspace runs its blocks as parallel tasks (each
# task makes its own BLAS calls). The runs that need it are counted, so that
# concurrent runs in one session restore the previous number of BLAS threads
# only when the last of them ends. `_blas_single!` returns whether it counted
# the run, for `_blas_restore`.
const _BLAS_LOCK = ReentrantLock()
const _BLAS_RUNS = Ref(0)
const _BLAS_SAVED = Ref(0)
function _blas_single!(ws)
    ws.nb == 1 && return false
    lock(_BLAS_LOCK) do
        if _BLAS_RUNS[] == 0
            _BLAS_SAVED[] = BLAS.get_num_threads()
            BLAS.set_num_threads(1)
        end
        _BLAS_RUNS[] += 1
    end
    return true
end
function _blas_restore(counted::Bool)
    counted || return nothing
    lock(_BLAS_LOCK) do
        _BLAS_RUNS[] -= 1
        _BLAS_RUNS[] == 0 && BLAS.set_num_threads(_BLAS_SAVED[])
    end
    return nothing
end

# f() with the number of BLAS threads of the session, also while other runs
# hold BLAS at one thread: the LAPACK inverse of the element stiffness in
# `precompute` depends on that number in the last bits. With concurrent runs
# in one session this raises the BLAS threads (and holds the lock) for the
# whole of `precompute` while the other runs make BLAS calls from their
# parallel regions. Whether changing the thread count with calls in flight is
# safe depends on OpenBLAS internals; no effect on the results or the speed of
# the other runs was measured, but such runs are better started as separate
# processes (README).
function _with_session_blas(f)
    lock(_BLAS_LOCK) do
        _BLAS_RUNS[] > 0 && BLAS.set_num_threads(_BLAS_SAVED[])
        try
            return f()
        finally
            _BLAS_RUNS[] > 0 && BLAS.set_num_threads(1)
        end
    end
end

"""
    rk4_step!(u, v, t, dt, N, h, pre, ws) -> (u, v)

One step of the classical fourth-order Runge–Kutta method, in place, with the
workspace `ws = Workspace(pre, N; nblocks)`. The stages are evaluated
straight into `ws.k1u, …, ws.k4v`, and the stage and final updates of the
state run on `ws.nb` parallel tasks as well (on contiguous ranges of the
arrays); the result does not depend on `ws.nb`. As for [`edg_rhs!`](@ref),
keep BLAS at one thread when `ws.nb > 1`. Afterwards `all(ws.fin)` tells
whether all entries of `u` and `v` are finite.
"""
function rk4_step!(u, v, t, dt, N, h, pre, ws)
    su = ws.su;  sv = ws.sv
    _edg_rhs!(ws.k1u, ws.k1v, ws, u, v, t, N, h, pre)
    _rk4_stage!(su, sv, u, v, 0.5*dt, ws.k1u, ws.k1v, ws);  _edg_rhs!(ws.k2u, ws.k2v, ws, su, sv, t + 0.5*dt, N, h, pre)
    _rk4_stage!(su, sv, u, v, 0.5*dt, ws.k2u, ws.k2v, ws);  _edg_rhs!(ws.k3u, ws.k3v, ws, su, sv, t + 0.5*dt, N, h, pre)
    _rk4_stage!(su, sv, u, v,     dt, ws.k3u, ws.k3v, ws);  _edg_rhs!(ws.k4u, ws.k4v, ws, su, sv, t +     dt, N, h, pre)
    _rk4_final!(u, v, dt/6.0, ws)
    return u, v
end

# su = u + c ku, sv = v + c kv
function _rk4_stage!(su, sv, u, v, c, ku, kv, ws)
    nb = ws.nb
    _foreach_block(nb) do b
        r = _chunk(length(su), nb, b);  x = view(su, r);  x .= view(u, r) .+ c .* view(ku, r)
        r = _chunk(length(sv), nb, b);  y = view(sv, r);  y .= view(v, r) .+ c .* view(kv, r)
    end
    return nothing
end

# u += c (k1 + 2 k2 + 2 k3 + k4), likewise v; ws.fin[b] tells whether the
# updated entries of block b are finite (the blow-up check of :source mode)
function _rk4_final!(u, v, c, ws)
    nb = ws.nb
    _foreach_block(nb) do b
        r = _chunk(length(u), nb, b);  x = view(u, r)
        x .= x .+ c .* (view(ws.k1u, r) .+ 2.0 .* view(ws.k2u, r) .+ 2.0 .* view(ws.k3u, r) .+ view(ws.k4u, r))
        r = _chunk(length(v), nb, b);  y = view(v, r)
        y .= y .+ c .* (view(ws.k1v, r) .+ 2.0 .* view(ws.k2v, r) .+ 2.0 .* view(ws.k3v, r) .+ view(ws.k4v, r))
        ws.fin[b] = all(isfinite, x) && all(isfinite, y)
    end
    return nothing
end

"""
    rk4_step(u, v, t, dt, N, h, pre; nblocks = nothing) -> (u, v)

One step of the classical fourth-order Runge–Kutta method (allocating
version of [`rk4_step!`](@ref) with `Workspace(pre, N; nblocks)`; `u`, `v`
are not modified). With more than one block, BLAS runs with one thread
during the step, as in [`solve`](@ref).
"""
function rk4_step(u, v, t, dt, N, h, pre; nblocks::Union{Nothing,Integer} = nothing)
    ws = Workspace(pre, N; nblocks = nblocks)
    counted = _blas_single!(ws)
    try
        return rk4_step!(copy(u), copy(v), t, dt, N, h, pre, ws)
    finally
        _blas_restore(counted)
    end
end

"""
    precompute(q_u, N; kwargs...) -> (pre, geo)

Build all element operators, face coefficients and data arrays used by
[`edg_rhs`](@ref) (`pre`), plus the quadrature geometry used for error
evaluation and diagnostics (`geo`). Keywords as in [`solve`](@ref).
"""
function precompute(q::Int, N::Int;
                    mixed::Bool = true, flux::Symbol = :upwind, flux_params = nothing,
                    sigma::Float64 = 0.0, boundary::Union{Symbol,NamedTuple} = :absorbing_2beta,
                    sigma_dirichlet::Float64 = sigma, kappa::Float64 = 0.0,
                    phase::Tuple{Float64,Float64} = (0.0, 0.0),
                    coefficients::Coefficients = paper_coefficients(),
                    mode::Symbol = :mms, kx::Int = 1, ky::Int = 1,
                    source::GaussianSource = GaussianSource(),
                    mass_solver::Symbol = :chebyshev, initial_profile = nothing)
    mass_solver in (:cg, :chebyshev) ||
        error("unknown mass_solver $mass_solver (expected :cg or :chebyshev)")
    initial_profile === nothing || mode == :initial_value ||
        error("initial_profile is only available in :initial_value mode")
    q_u = q
    q_v = mixed ? q_u - 1 : q_u
    qu1 = q_u + 1
    qv1 = q_v + 1
    h   = 1.0 / N
    omega = 2π * sqrt(kx^2 + ky^2)

    fp = flux_params === nothing ? flux_parameters(flux) :
         (theta = Float64(flux_params.theta), tau = Float64(flux_params.tau),
          lambda_u = Float64(flux_params.lambda_u), lambda_v = Float64(flux_params.lambda_v))
    bc = _boundary_sides(boundary, sigma, sigma_dirichlet, kappa)
    # γ, σ of the uniform legacy kinds, read only by the reference RHS
    # (test/reference_rhs.jl); NaN for every other boundary
    legacy    = boundary in (:absorbing, :dirichlet)
    gamma_abc = legacy ? bc.bot.gamma_abc : NaN
    px, py    = phase

    cC = coefficients.c;  cxC = coefficients.c_x;  cyC = coefficients.c_y
    alphaC = coefficients.alpha;  alphaxC = coefficients.alpha_x;  alphayC = coefficients.alpha_y
    betaC = coefficients.beta

    # ---- 1D bases at the standard (q_u+1)-point Gauss rule ----
    z, w = GaussQCofs(qu1)
    W2d  = w * w'

    P_u    = zeros(qu1, qu1);  S_u    = zeros(qu1, qu1)
    Ebl_u  = zeros(1, qu1);    Ebr_u  = zeros(1, qu1)
    E1bl_u = zeros(1, qu1);    E1br_u = zeros(1, qu1)
    for d in 1:qu1
        P_u[:, d]    = JacobiP(z, 0, 0, d-1)
        S_u[:, d]    = GradJacobiP(z, 0, 0, d-1)
        Ebl_u[1, d]  = JacobiP(-1.0, 0, 0, d-1)
        Ebr_u[1, d]  = JacobiP( 1.0, 0, 0, d-1)
        E1bl_u[1, d] = GradJacobiP(-1.0, 0, 0, d-1)
        E1br_u[1, d] = GradJacobiP( 1.0, 0, 0, d-1)
    end
    # v-space: the first q_v+1 Legendre modes
    P_v    = P_u[:, 1:qv1]
    Ebl_v  = Ebl_u[:, 1:qv1];    Ebr_v  = Ebr_u[:, 1:qv1]
    E1bl_v = E1bl_u[:, 1:qv1];   E1br_v = E1br_u[:, 1:qv1]

    # ---- high-order rule for the source and the nonlinear mass ----
    q_high = max(Int(ceil(3*q_v/2) + 1), q_u + 1)
    z_h, w_h = GaussQCofs(q_high)
    W2d_h    = w_h * w_h'
    P_h_v = zeros(q_high, qv1)
    for d in 1:qv1
        P_h_v[:, d] = JacobiP(z_h, 0, 0, d-1)
    end
    Phi2d_h_v = kron(P_h_v, P_h_v)

    # ---- 2D tensor-product derivative bases ----
    Phi_xi_u  = kron(P_u, S_u)
    Phi_eta_u = kron(S_u, P_u)
    S_v       = S_u[:, 1:qv1]

    # ---- quadrature coordinates per element row/column ----
    xloc_all   = zeros(N, qu1)
    yloc_all   = zeros(N, qu1)
    xloc_h_all = zeros(N, q_high)
    yloc_h_all = zeros(N, q_high)
    for i in 1:N
        xloc_all[i, :]   = ((i-1)*h + i*h .+ h .* z)  ./ 2
        yloc_all[i, :]   = xloc_all[i, :]
        xloc_h_all[i, :] = ((i-1)*h + i*h .+ h .* z_h) ./ 2
        yloc_h_all[i, :] = xloc_h_all[i, :]
    end

    # ---- volume terms: c²W and αW at the points of every element (point,
    # element, point; rows y, columns x), applied matrix-free in rhs.jl, and the
    # inverse of the pinned u-stiffness, upper triangle packed by columns
    # (entry (i, j), i ≤ j, in row j(j-1)/2 + i; element in the column) ----
    Cw    = zeros(qu1, N * N, qu1)
    Aw    = zeros(qu1, N * N, qu1)
    nu2   = qu1^2
    Ainv_u = zeros((nu2 * (nu2 + 1)) >> 1, N * N)
    for j in 1:N, k in 1:N
        # rows vary in y, columns in x
        Xq = repeat(xloc_all[k, :]', qu1, 1)
        Yq = repeat(yloc_all[j, :],  1, qu1)

        cw = vec(cC.(Xq, Yq).^2  .* W2d)
        aw = vec(alphaC.(Xq, Yq) .* W2d)

        Cw[:, j + (k - 1) * N, :] = reshape(cw, qu1, qu1)
        Aw[:, j + (k - 1) * N, :] = reshape(aw, qu1, qu1)

        # c²-stiffness with the constant mode pinned (elementwise mean condition)
        Mj = Phi_xi_u' * (cw .* Phi_xi_u) .+ Phi_eta_u' * (cw .* Phi_eta_u)
        Mj[1, 1] += 1.0
        Aj = inv(cholesky(Symmetric(Mj)))
        e = j + (k - 1) * N
        for jj in 1:nu2, ii in 1:jj
            Ainv_u[((jj * (jj - 1)) >> 1) + ii, e] = Aj[ii, jj]
        end
    end

    # ---- face coefficients (N×N×(q_u+1), one entry per face quadrature point) ----
    faces = Dict{Symbol,NamedTuple}()
    for (name, xyf) in ((:bot, (j, k) -> (xloc_all[k, :], fill((j-1)*h, qu1))),
                        (:top, (j, k) -> (xloc_all[k, :], fill(j*h, qu1))),
                        (:lft, (j, k) -> (fill((k-1)*h, qu1), yloc_all[j, :])),
                        (:rgt, (j, k) -> (fill(k*h, qu1), yloc_all[j, :])))
        c2 = zeros(N, N, qu1);  af = zeros(N, N, qu1)
        cf = zeros(N, N, qu1);  bf = zeros(N, N, qu1)
        for j in 1:N, k in 1:N
            xf, yf = xyf(j, k)
            cvals = cC.(xf, yf)
            cf[j, k, :] = cvals
            c2[j, k, :] = cvals.^2
            af[j, k, :] = alphaC.(xf, yf)
            bf[j, k, :] = betaC.(xf, yf)
        end
        faces[name] = (c2 = c2, af = af, cf = cf, bf = bf)
    end

    # ---- manufactured-solution data at the high-order points ----
    # Φ, ∇·(c²∇Φ), ∇·(α∇Φ), β
    mms_SP   = zeros(N, N, q_high, q_high)
    mms_Cdiv = zeros(N, N, q_high, q_high)
    mms_Ddiv = zeros(N, N, q_high, q_high)
    mms_BF   = zeros(N, N, q_high, q_high)
    for j in 1:N, k in 1:N
        Xh = repeat(xloc_h_all[k, :]', q_high, 1)
        Yh = repeat(yloc_h_all[j, :],  1, q_high)

        if initial_profile === nothing
            sp    =  cos.(2π*kx .* Xh .+ px) .* sin.(2π*ky .* Yh .+ py)
            sp_x  = -2π*kx .* sin.(2π*kx .* Xh .+ px) .* sin.(2π*ky .* Yh .+ py)
            sp_y  =  2π*ky .* cos.(2π*kx .* Xh .+ px) .* cos.(2π*ky .* Yh .+ py)
            sp_xx = -(2π*kx)^2 .* sp
            sp_yy = -(2π*ky)^2 .* sp
        else
            ip = initial_profile
            sp = ip.u0.(Xh, Yh);  sp_x = ip.u0_x.(Xh, Yh);  sp_y = ip.u0_y.(Xh, Yh)
            sp_xx = ip.u0_xx.(Xh, Yh);  sp_yy = ip.u0_yy.(Xh, Yh)
        end

        cf  = cC.(Xh, Yh);   c2  = cf.^2
        cx  = cxC.(Xh, Yh);  cy  = cyC.(Xh, Yh)
        c2x = 2 .* cf .* cx; c2y = 2 .* cf .* cy
        af  = alphaC.(Xh, Yh)
        ax  = alphaxC.(Xh, Yh); ay = alphayC.(Xh, Yh)

        mms_SP[j, k, :, :]   = sp
        mms_Cdiv[j, k, :, :] = c2x .* sp_x .+ c2y .* sp_y .+ c2 .* (sp_xx .+ sp_yy)
        mms_Ddiv[j, k, :, :] = ax .* sp_x .+ ay .* sp_y .+ af .* (sp_xx .+ sp_yy)
        mms_BF[j, k, :, :]   = betaC.(Xh, Yh)
    end

    # ---- Φ and ∇Φ on the physical boundary ----
    bc_bot_sp  = zeros(N, qu1);  bc_bot_spy = zeros(N, qu1)
    bc_top_sp  = zeros(N, qu1);  bc_top_spy = zeros(N, qu1)
    bc_lft_sp  = zeros(N, qu1);  bc_lft_spx = zeros(N, qu1)
    bc_rgt_sp  = zeros(N, qu1);  bc_rgt_spx = zeros(N, qu1)
    for k in 1:N
        xq = xloc_all[k, :]
        bc_bot_sp[k, :]  =  cos.(2π*kx .* xq .+ px) .* sin(2π*ky*0 + py)          # y = 0
        bc_bot_spy[k, :] =  2π*ky .* cos.(2π*kx .* xq .+ px) .* cos(2π*ky*0 + py)
        bc_top_sp[k, :]  =  cos.(2π*kx .* xq .+ px) .* sin(2π*ky*1 + py)          # y = 1
        bc_top_spy[k, :] =  2π*ky .* cos.(2π*kx .* xq .+ px) .* cos(2π*ky*1 + py)
    end
    for j in 1:N
        yq = yloc_all[j, :]
        bc_lft_sp[j, :]  =  cos(2π*kx*0 + px) .* sin.(2π*ky .* yq .+ py)          # x = 0
        bc_lft_spx[j, :] = -2π*kx .* sin(2π*kx*0 + px) .* sin.(2π*ky .* yq .+ py)
        bc_rgt_sp[j, :]  =  cos(2π*kx*1 + px) .* sin.(2π*ky .* yq .+ py)          # x = 1
        bc_rgt_spx[j, :] = -2π*kx .* sin(2π*kx*1 + px) .* sin.(2π*ky .* yq .+ py)
    end
    if initial_profile !== nothing
        ip = initial_profile
        for k in 1:N
            xq = xloc_all[k, :]
            bc_bot_sp[k, :] = ip.u0.(xq, 0.0);  bc_bot_spy[k, :] = ip.u0_y.(xq, 0.0)
            bc_top_sp[k, :] = ip.u0.(xq, 1.0);  bc_top_spy[k, :] = ip.u0_y.(xq, 1.0)
        end
        for j in 1:N
            yq = yloc_all[j, :]
            bc_lft_sp[j, :] = ip.u0.(0.0, yq);  bc_lft_spx[j, :] = ip.u0_x.(0.0, yq)
            bc_rgt_sp[j, :] = ip.u0.(1.0, yq);  bc_rgt_spx[j, :] = ip.u0_x.(1.0, yq)
        end
    end

    # ---- c² and β at the standard points (errors, diagnostics) ----
    err_c2   = zeros(N, N, qu1, qu1)
    err_beta = zeros(N, N, qu1, qu1)
    for j in 1:N, k in 1:N
        Xq = repeat(xloc_all[k, :]', qu1, 1)
        Yq = repeat(yloc_all[j, :],  1, qu1)
        err_c2[j, k, :, :]   = cC.(Xq, Yq).^2
        err_beta[j, k, :, :] = betaC.(Xq, Yq)
    end

    # ---- source mode: no manufactured data (u₀ = 0, homogeneous boundary) ----
    src_on   = mode == :source
    src_proj = zeros(N, N, qv1, qv1)
    if src_on
        for A in (mms_SP, mms_Cdiv, mms_Ddiv,
                  bc_bot_sp, bc_bot_spy, bc_top_sp, bc_top_spy,
                  bc_lft_sp, bc_lft_spx, bc_rgt_sp, bc_rgt_spx)
            fill!(A, 0.0)
        end
        for j in 1:N, k in 1:N
            Xh = repeat(xloc_h_all[k, :]', q_high, 1);  Yh = repeat(yloc_h_all[j, :], 1, q_high)
            g  = source.amplitude .* exp.(-((Xh .- source.x0[1]).^2 .+ (Yh .- source.x0[2]).^2) ./ source.r0^2)
            src_proj[j, k, :, :] = P_h_v' * (W2d_h .* g) * P_h_v
        end
    end
    src_time(t) = (src_on && t < source.Ts) ? sin(source.omega*t) * sin(π*t/source.Ts)^2 : 0.0

    fb = faces[:bot];  ft = faces[:top];  fl = faces[:lft];  fr = faces[:rgt]
    pre = (
        qu1 = qu1,  qv1 = qv1,  q_high = q_high,  omega = omega,
        w   = w,
        P_u = P_u,  P_v = P_v,  S_u = S_u,  S_v = S_v,
        Ebl_u = Ebl_u,  Ebr_u = Ebr_u,  E1bl_u = E1bl_u,  E1br_u = E1br_u,
        Ebl_v = Ebl_v,  Ebr_v = Ebr_v,  E1bl_v = E1bl_v,  E1br_v = E1br_v,
        W2d_h = W2d_h,  P_h_v = P_h_v,  Phi2d_h_v = Phi2d_h_v,
        Cw = Cw,  Aw = Aw,  Ainv_u = Ainv_u,
        bot_c2 = fb.c2,  bot_af = fb.af,  bot_cf = fb.cf,  bot_bf = fb.bf,
        top_c2 = ft.c2,  top_af = ft.af,  top_cf = ft.cf,  top_bf = ft.bf,
        lft_c2 = fl.c2,  lft_af = fl.af,  lft_cf = fl.cf,  lft_bf = fl.bf,
        rgt_c2 = fr.c2,  rgt_af = fr.af,  rgt_cf = fr.cf,  rgt_bf = fr.bf,
        mms_SP = mms_SP,  mms_Cdiv = mms_Cdiv,  mms_Ddiv = mms_Ddiv,  mms_BF = mms_BF,
        bc_bot_sp = bc_bot_sp,  bc_bot_spy = bc_bot_spy,
        bc_top_sp = bc_top_sp,  bc_top_spy = bc_top_spy,
        bc_lft_sp = bc_lft_sp,  bc_lft_spx = bc_lft_spx,
        bc_rgt_sp = bc_rgt_sp,  bc_rgt_spx = bc_rgt_spx,
        theta = fp.theta,  tau = fp.tau,  lambda_u = fp.lambda_u,  lambda_v = fp.lambda_v,
        gamma_abc = gamma_abc,  eta_abc = sqrt(1.0 - gamma_abc^2),  sigma = legacy ? bc.bot.sigma : NaN,
        bc = bc,
        src_on = src_on,  src_proj = src_proj,  src_time = src_time,
        mms_on = mode == :mms,
        mass_solver = mass_solver,
    )
    pre = merge(pre, _rhs_data(pre, N))
    geo = (S_u = S_u, W2d = W2d, xloc = xloc_all, yloc = yloc_all,
           err_c2 = err_c2, err_beta = err_beta)
    return pre, geo
end

# Data of the batched RHS (src/rhs.jl), derived from the element-by-element
# arrays above:
# - face coefficients by face: horizontal faces (point, face row 1..N+1, column),
#   vertical faces (point, row, face column 1..N+1);
# - boundary data by point: (point, column) on y = 0, 1 and (point, row) on x = 0, 1;
# - Π(X) = P_h_v' (W ∘ X) P_h_v for X = Φ, βΦ², ∇·(c²∇Φ), ∇·(α∇Φ) per element
#   (element-last), so that the manufactured source is a combination of them;
# - β at the high-order points (element-last, and as (point, element, point)
#   with the weights as (point, 1, point) for the Chebyshev solver) and P_h_v'
#   for the mass solve;
# - the transposes of the 1D values and derivatives at the standard points for
#   the volume terms.
function _rhs_data(pre, N)
    nq = pre.qu1;  nv = pre.qv1;  qh = pre.q_high
    hface(bot, top) = (A = zeros(nq, N + 1, N);
                       for k in 1:N, j in 1:N; A[:, j, k] = bot[j, k, :]; end;
                       for k in 1:N; A[:, N + 1, k] = top[N, k, :]; end; A)
    vface(lft, rgt) = (A = zeros(nq, N, N + 1);
                       for k in 1:N, j in 1:N; A[:, j, k] = lft[j, k, :]; end;
                       for j in 1:N; A[:, j, N + 1] = rgt[j, N, :]; end; A)
    proj(X) = (Pi = zeros(nv, nv, N, N);
               for k in 1:N, j in 1:N
                   Pi[:, :, j, k] = pre.P_h_v' * (pre.W2d_h .* X[j, k, :, :]) * pre.P_h_v
               end; Pi)
    elast(X) = permutedims(X, (3, 4, 1, 2))
    (hc2 = hface(pre.bot_c2, pre.top_c2), haf = hface(pre.bot_af, pre.top_af),
     hcf = hface(pre.bot_cf, pre.top_cf), hbf = hface(pre.bot_bf, pre.top_bf),
     vc2 = vface(pre.lft_c2, pre.rgt_c2), vaf = vface(pre.lft_af, pre.rgt_af),
     vcf = vface(pre.lft_cf, pre.rgt_cf), vbf = vface(pre.lft_bf, pre.rgt_bf),
     bcB_sp = permutedims(pre.bc_bot_sp), bcB_spy = permutedims(pre.bc_bot_spy),
     bcT_sp = permutedims(pre.bc_top_sp), bcT_spy = permutedims(pre.bc_top_spy),
     bcL_sp = permutedims(pre.bc_lft_sp), bcL_spx = permutedims(pre.bc_lft_spx),
     bcR_sp = permutedims(pre.bc_rgt_sp), bcR_spx = permutedims(pre.bc_rgt_spx),
     PiSP = proj(pre.mms_SP), PiBSP2 = proj(pre.mms_BF .* pre.mms_SP .^ 2),
     PiCdiv = proj(pre.mms_Cdiv), PiDdiv = proj(pre.mms_Ddiv),
     src_projE = elast(pre.src_proj), beta_hE = elast(pre.mms_BF),
     beta_hq = permutedims(reshape(elast(pre.mms_BF), qh, qh, N * N), (1, 3, 2)),
     W_hq = reshape(copy(pre.W2d_h), qh, 1, qh),
     P_h_vT = permutedims(pre.P_h_v), mass_tol = 1e-14,
     P_uT = permutedims(pre.P_u), S_uT = permutedims(pre.S_u),
     P_vT = permutedims(pre.P_v), S_vT = permutedims(pre.S_v))
end

# L² errors of u, v, ∇u and the energy error against u = cos(ωT) Φ at time T.
# The solver stores ũ = u - Φ, so it is compared with (cos(ωT) - 1) Φ.
function _mms_errors(u, v, T, N, h, pre, geo, kx, ky, phase = (0.0, 0.0))
    px, py = phase
    P_u = pre.P_u;  P_v = pre.P_v;  S_u = geo.S_u;  W2d = geo.W2d;  qu1 = pre.qu1
    erru = 0.0;  errv = 0.0;  errgradu = 0.0;  errE = 0.0
    ct_T = cos(pre.omega * T)
    st_T = sin(pre.omega * T)

    for j in 1:N, k in 1:N
        Xq = repeat(geo.xloc[k, :]', qu1, 1)
        Yq = repeat(geo.yloc[j, :],  1, qu1)

        sp    =  cos.(2π*kx .* Xq .+ px) .* sin.(2π*ky .* Yq .+ py)
        sp_x  = -2π*kx .* sin.(2π*kx .* Xq .+ px) .* sin.(2π*ky .* Yq .+ py)
        sp_y  =  2π*ky .* cos.(2π*kx .* Xq .+ px) .* cos.(2π*ky .* Yq .+ py)

        utrue  =  (ct_T - 1.0) .* sp
        vtrue  = -pre.omega * st_T .* sp
        uxtrue =  (ct_T - 1.0) .* sp_x
        uytrue =  (ct_T - 1.0) .* sp_y

        uloc = u[j, k, :, :];  vloc = v[j, k, :, :]
        u_h  = P_u * uloc * P_u'
        v_h  = P_v * vloc * P_v'
        ux_h = (2/h) .* (P_u * uloc * S_u')
        uy_h = (2/h) .* (S_u * uloc * P_u')
        c2q  = geo.err_c2[j, k, :, :]

        erru     += sum(W2d .* (utrue  .- u_h).^2)
        errv     += sum(W2d .* (vtrue  .- v_h).^2)
        errgradu += sum(W2d .* ((uxtrue .- ux_h).^2 .+ (uytrue .- uy_h).^2))
        errE     += sum(W2d .* (0.5 .* c2q .* ((uxtrue .- ux_h).^2 .+ (uytrue .- uy_h).^2)
                                .+ 0.5 .* (vtrue .- v_h).^2))
    end
    s = h / 2
    return (erru = s * sqrt(erru), errv = s * sqrt(errv),
            errgradu = s * sqrt(errgradu), errE = s * sqrt(errE))
end

"""
    difference_norms(A, B) -> (erru, errv, errE)

L² norms of `u_A - u_B`, `v_A - v_B` and the energy norm
`‖e‖_en² = ½‖c∇_h e_u‖² + ½‖e_v‖²` of the difference of two final states
returned by `solve(...; return_state = true)` on the same mesh.
"""
function difference_norms(A, B)
    N = B.N;  h = B.h;  P_u = B.P_u;  P_v = B.P_v;  S_u = B.S_u;  W2d = B.W2d
    erru = 0.0;  errv = 0.0;  errE = 0.0
    for j in 1:N, k in 1:N
        eu = A.u[j, k, :, :] .- B.u[j, k, :, :]
        ev = A.v[j, k, :, :] .- B.v[j, k, :, :]
        du  = P_u * eu * P_u'
        dux = (2 / h) .* (P_u * eu * S_u')
        duy = (2 / h) .* (S_u * eu * P_u')
        dv  = P_v * ev * P_v'
        c2q = B.err_c2[j, k, :, :]
        erru += sum(W2d .* du .^ 2)
        errv += sum(W2d .* dv .^ 2)
        errE += sum(W2d .* (0.5 .* c2q .* (dux .^ 2 .+ duy .^ 2) .+ 0.5 .* dv .^ 2))
    end
    s = h / 2
    return (erru = s * sqrt(erru), errv = s * sqrt(errv), errE = s * sqrt(errE))
end

# C = A B for the small matrices of the energy diagnostics, summed as `*`
# sums them: Julia's own formulas (no FMA) when C, A and B are all 2×2 or all
# 3×3, otherwise as BLAS (`_mul_loops!`). Plain loops, because small BLAS
# calls from parallel tasks are slower than the serial loop.
function _mul_small!(C, A, B)
    n = size(A, 1)
    if (n == 2 || n == 3) && size(C) == size(A) == size(B) == (n, n)
        @inbounds for j in 1:n, i in 1:n
            s = A[i, 1] * B[1, j] + A[i, 2] * B[2, j]
            n == 3 && (s = s + A[i, 3] * B[3, j])
            C[i, j] = s
        end
        return C
    end
    return _mul_loops!(C, A, B)
end

# Element values of the energy diagnostics, and work arrays per block.
function _diag_workspace(pre, N, ws)
    nq = pre.qu1;  nu = pre.qu1;  nv = pre.qv1
    (E = zeros(N * N), vmax = zeros(N * N), ndmin = zeros(N * N),
     blk = [(U = zeros(nu, nu), V = zeros(nv, nv), Tu = zeros(nq, nu), Tv = zeros(nq, nv),
             ux = zeros(nq, nq), uy = zeros(nq, nq), vh = zeros(nq, nq),
             c2q = zeros(nq, nq), bq = zeros(nq, nq), Z = zeros(nq, nq)) for _ in 1:ws.nb])
end

# On every element: the Westervelt energy (h²/4) Σ W (½c²|∇u_h|² + (½ - ⅔βv_h)v_h²)
# at the points, max|v_h| and min(1 - 2βv_h), on the blocks of ws in parallel;
# bitwise the values of the serial expressions (h²/4) * sum(W2d .* (...)) with
# ∇u_h = (2/h)(P_u U S_u', S_u U P_u') and v_h = P_v V P_v'.
function _energy_diag!(dg, u, v, pre, geo, N, h, ws)
    W2d = geo.W2d
    _foreach_block(ws.nb) do b
        B = dg.blk[b]
        U = B.U;  V = B.V;  Tu = B.Tu;  Tv = B.Tv;  ux = B.ux;  uy = B.uy;  vh = B.vh
        c2q = B.c2q;  bq = B.bq;  Z = B.Z
        for e in ws.eblk[b]
            j = (e - 1) % N + 1;  k = (e - 1) ÷ N + 1
            U .= view(u, j, k, :, :);  V .= view(v, j, k, :, :)
            _mul_small!(Tu, pre.P_u, U);  _mul_small!(ux, Tu, pre.S_uT);  ux .= (2/h) .* ux
            _mul_small!(Tu, pre.S_u, U);  _mul_small!(uy, Tu, pre.P_uT);  uy .= (2/h) .* uy
            _mul_small!(Tv, pre.P_v, V);  _mul_small!(vh, Tv, pre.P_vT)
            c2q .= view(geo.err_c2, j, k, :, :);  bq .= view(geo.err_beta, j, k, :, :)
            Z .= W2d .* (0.5 .* c2q .* (ux.^2 .+ uy.^2) .+ (0.5 .- (2/3) .* bq .* vh) .* vh.^2)
            dg.E[e] = (h^2/4) * sum(Z)
            Z .= abs.(vh);  dg.vmax[e] = maximum(Z)
            Z .= 1 .- 2 .* bq .* vh;  dg.ndmin[e] = minimum(Z)
        end
    end
    return nothing
end

# Source mode: RK4 with receiver time series, line slices, energy diagnostics
# and full-field snapshots of v_h.
function _run_source(u, v, dt, nsteps, N, h, pre, geo, ws;
                     record_every, slice_y, receivers, snap_times)
    qu1 = pre.qu1;  qv1 = pre.qv1;  P_v = pre.P_v
    xloc_all = geo.xloc;  yloc_all = geo.yloc
    t = 0.0

    # modal basis of v evaluated at a physical point inside element (j, k)
    locate(s) = (i = clamp(Int(ceil(s*N)), 1, N); (i, clamp(2*(s*N - (i-1)) - 1, -1.0, 1.0)))
    vrow(r)   = reshape([JacobiP(r, 0, 0, d-1) for d in 1:qv1], 1, qv1)

    j0, eta0 = locate(slice_y)
    Pveta0   = vrow(eta0)
    slice_x  = vcat([xloc_all[k, :] for k in 1:N]...)
    slicef(vv) = vcat([P_v * (Pveta0 * vv[j0, k, :, :])' for k in 1:N]...)

    rcv = map(receivers) do (xr, yr)
        kr, xir  = locate(xr)
        jr, etar = locate(yr)
        (jr, kr, vrow(etar), vrow(xir))
    end
    recvf(vv) = [(r[3] * vv[r[1], r[2], :, :] * r[4]')[1] for r in rcv]

    # Westervelt energy ∫ ½c²|∇u|² + (½ - ⅔βv)v², max|v_h| and min(1 - 2βv_h):
    # the values of the elements on the blocks of ws in parallel
    # (`_energy_diag!`), then accumulated serially in the order row j, column k
    dg = _diag_workspace(pre, N, ws)
    function diagf(uu, vv)
        _energy_diag!(dg, uu, vv, pre, geo, N, h, ws)
        E = 0.0;  vmax = 0.0;  ndmin = Inf
        for j in 1:N, k in 1:N
            e = j + (k - 1) * N
            E += dg.E[e];  vmax = max(vmax, dg.vmax[e]);  ndmin = min(ndmin, dg.ndmin[e])
        end
        return E, vmax, ndmin
    end
    vgrid(vv) = (G = zeros(N*qu1, N*qu1);
                 for j in 1:N, k in 1:N
                     G[(j-1)*qu1+1:j*qu1, (k-1)*qu1+1:k*qu1] = P_v * vv[j,k,:,:] * P_v'
                 end; G)

    rec_t = Float64[];  rec_slice = Vector{Float64}[]
    rec_E = Float64[];  rec_vmax = Float64[];  rec_nd = Float64[]
    rt_t  = Float64[];  rt_v = Vector{Float64}[]
    snaps = Tuple{Float64, Matrix{Float64}}[]
    snap_left = sort(collect(snap_times))
    for it in 1:nsteps
        rk4_step!(u, v, t, dt, N, h, pre, ws)
        t += dt
        all(ws.fin) || error("blow-up at t=$t")    # u, v finite (checked in `_rk4_final!`)
        push!(rt_t, t);  push!(rt_v, recvf(v))
        if it % record_every == 0 || it == nsteps
            E, vmax, ndmin = diagf(u, v)
            push!(rec_t, t);  push!(rec_slice, vec(slicef(v)))
            push!(rec_E, E);  push!(rec_vmax, vmax);  push!(rec_nd, ndmin)
        end
        if !isempty(snap_left) && t >= snap_left[1]
            push!(snaps, (t, vgrid(v)));  popfirst!(snap_left)
        end
    end
    grid_x = vcat([xloc_all[k, :] for k in 1:N]...)
    grid_y = vcat([yloc_all[j, :] for j in 1:N]...)
    return (t = rec_t, slice_x = slice_x, slices = hcat(rec_slice...),
            energy = rec_E, vmax = rec_vmax, ndmin = rec_nd,
            recv_t = rt_t, recv = permutedims(hcat(rt_v...)),
            receivers = receivers, snaps = snaps,
            grid_x = grid_x, grid_y = grid_y, dt = dt)
end
