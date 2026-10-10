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
#                     paper's Remark 2.1 (the default of v0.1 and v0.2);
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
  `‖e‖_E² = ½‖c∇_h e_u‖² + ½‖e_v‖²` at time `T`.
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
- `coefficients = paper_coefficients()`: see [`Coefficients`](@ref).
- `kx = 1, ky = 1`: wave numbers of Φ.
- `dt_fac = 10`.
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
               return_state::Bool = false,
               source::GaussianSource = GaussianSource(),
               record_every::Int = 20,
               slice_y::Float64 = 0.5,
               receivers = [(0.45, 0.5), (0.7, 0.5), (0.85, 0.5)],
               snap_times = Float64[])

    mode in (:mms, :initial_value, :source) ||
        error("unknown mode $mode (expected :mms, :initial_value or :source)")
    pre, geo = precompute(q, N; mixed = mixed, flux = flux, flux_params = flux_params,
                          sigma = sigma, boundary = boundary, sigma_dirichlet = sigma_dirichlet,
                          kappa = kappa, phase = phase, coefficients = coefficients,
                          mode = mode, kx = kx, ky = ky, source = source)

    h  = 1.0 / N
    u  = zeros(N, N, pre.qu1, pre.qu1)     # ũ = u - u₀
    v  = zeros(N, N, pre.qv1, pre.qv1)
    dt = h / (dt_fac * q^2)
    nsteps = round(Int, T / dt)
    dt = T / nsteps

    mode == :source && return _run_source(u, v, dt, nsteps, N, h, pre, geo;
                                          record_every = record_every, slice_y = slice_y,
                                          receivers = receivers, snap_times = snap_times)

    t = 0.0
    ws = Workspace(pre, N)
    for _ in 1:nsteps
        rk4_step!(u, v, t, dt, N, h, pre, ws)
        t += dt
    end

    if return_state
        return (u = u, v = v, N = N, h = h,
                P_u = pre.P_u, P_v = pre.P_v, S_u = geo.S_u, W2d = geo.W2d, err_c2 = geo.err_c2)
    end
    mode == :mms || error("errors are only available in :mms mode; use return_state = true")
    return _mms_errors(u, v, T, N, h, pre, geo, kx, ky, phase)
end

"""
    rk4_step!(u, v, t, dt, N, h, pre, ws) -> (u, v)

One step of the classical fourth-order Runge–Kutta method, in place, with the
workspace `ws = Workspace(pre, N)`.
"""
function rk4_step!(u, v, t, dt, N, h, pre, ws)
    su = ws.su;  sv = ws.sv
    function stage!(ku, kv, tt)
        a, b = edg_rhs!(ws, su, sv, tt, N, h, pre)
        ku .= a;  kv .= b
    end
    su .= u;  sv .= v;                                         stage!(ws.k1u, ws.k1v, t)
    su .= u .+ 0.5*dt .* ws.k1u;  sv .= v .+ 0.5*dt .* ws.k1v;  stage!(ws.k2u, ws.k2v, t + 0.5*dt)
    su .= u .+ 0.5*dt .* ws.k2u;  sv .= v .+ 0.5*dt .* ws.k2v;  stage!(ws.k3u, ws.k3v, t + 0.5*dt)
    su .= u .+     dt .* ws.k3u;  sv .= v .+     dt .* ws.k3v;  stage!(ws.k4u, ws.k4v, t +     dt)
    u .= u .+ (dt/6.0) .* (ws.k1u .+ 2.0 .* ws.k2u .+ 2.0 .* ws.k3u .+ ws.k4u)
    v .= v .+ (dt/6.0) .* (ws.k1v .+ 2.0 .* ws.k2v .+ 2.0 .* ws.k3v .+ ws.k4v)
    return u, v
end

"""
    rk4_step(u, v, t, dt, N, h, pre) -> (u, v)

One step of the classical fourth-order Runge–Kutta method (allocating
version of [`rk4_step!`](@ref); `u`, `v` are not modified).
"""
rk4_step(u, v, t, dt, N, h, pre) = rk4_step!(copy(u), copy(v), t, dt, N, h, pre, Workspace(pre, N))

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
                    source::GaussianSource = GaussianSource())
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
    Phi_xi_v  = kron(P_v, S_v)
    Phi_eta_v = kron(S_v, P_v)

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

    # ---- volume operators: (c²∇φ_v, ∇u), (α∇φ_v, ∇v), pinned u-stiffness ----
    Mc2_uv = Matrix{Matrix{Float64}}(undef, N, N)
    Ma_vv  = Matrix{Matrix{Float64}}(undef, N, N)
    R_ut   = Matrix{UpperTriangular{Float64, Matrix{Float64}}}(undef, N, N)
    for j in 1:N, k in 1:N
        # rows vary in y, columns in x
        Xq = repeat(xloc_all[k, :]', qu1, 1)
        Yq = repeat(yloc_all[j, :],  1, qu1)

        cw = vec(cC.(Xq, Yq).^2  .* W2d)
        aw = vec(alphaC.(Xq, Yq) .* W2d)

        Mc2_uv[j, k] = Phi_xi_v' * (cw .* Phi_xi_u) .+ Phi_eta_v' * (cw .* Phi_eta_u)
        Ma_vv[j, k]  = Phi_xi_v' * (aw .* Phi_xi_v) .+ Phi_eta_v' * (aw .* Phi_eta_v)

        # c²-stiffness with the constant mode pinned (elementwise mean condition)
        Mj = Phi_xi_u' * (cw .* Phi_xi_u) .+ Phi_eta_u' * (cw .* Phi_eta_u)
        Mj[1, 1] += 1.0
        R_ut[j, k] = cholesky(Symmetric(Mj)).U
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

        sp    =  cos.(2π*kx .* Xh .+ px) .* sin.(2π*ky .* Yh .+ py)
        sp_x  = -2π*kx .* sin.(2π*kx .* Xh .+ px) .* sin.(2π*ky .* Yh .+ py)
        sp_y  =  2π*ky .* cos.(2π*kx .* Xh .+ px) .* cos.(2π*ky .* Yh .+ py)
        sp_xx = -(2π*kx)^2 .* sp
        sp_yy = -(2π*ky)^2 .* sp

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
        P_u = P_u,  P_v = P_v,
        Ebl_u = Ebl_u,  Ebr_u = Ebr_u,  E1bl_u = E1bl_u,  E1br_u = E1br_u,
        Ebl_v = Ebl_v,  Ebr_v = Ebr_v,  E1bl_v = E1bl_v,  E1br_v = E1br_v,
        W2d_h = W2d_h,  P_h_v = P_h_v,  Phi2d_h_v = Phi2d_h_v,
        Mc2_uv = Mc2_uv,  Ma_vv = Ma_vv,  R_ut = R_ut,
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
# - β at the high-order points (element-last) and P_h_v' for the mass solve.
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
     P_h_vT = permutedims(pre.P_h_v), mass_tol = 1e-14)
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
`‖e‖_E² = ½‖c∇_h e_u‖² + ½‖e_v‖²` of the difference of two final states
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

# Source mode: RK4 with receiver time series, line slices, energy diagnostics
# and full-field snapshots of v_h.
function _run_source(u, v, dt, nsteps, N, h, pre, geo;
                     record_every, slice_y, receivers, snap_times)
    qu1 = pre.qu1;  qv1 = pre.qv1
    P_u = pre.P_u;  P_v = pre.P_v;  S_u = geo.S_u;  W2d = geo.W2d
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

    # Westervelt energy ∫ ½c²|∇u|² + (½ - ⅔βv)v², max|v_h| and min(1 - 2βv_h)
    function diagf(uu, vv)
        E = 0.0;  vmax = 0.0;  ndmin = Inf
        for j in 1:N, k in 1:N
            ux = (2/h) .* (P_u * uu[j,k,:,:] * S_u');  uy = (2/h) .* (S_u * uu[j,k,:,:] * P_u')
            vh = P_v * vv[j,k,:,:] * P_v'
            c2q = geo.err_c2[j,k,:,:];  bq = geo.err_beta[j,k,:,:]
            E += (h^2/4) * sum(W2d .* (0.5 .* c2q .* (ux.^2 .+ uy.^2)
                                       .+ (0.5 .- (2/3) .* bq .* vh) .* vh.^2))
            vmax  = max(vmax, maximum(abs.(vh)));  ndmin = min(ndmin, minimum(1 .- 2 .* bq .* vh))
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
    ws = Workspace(pre, N)
    for it in 1:nsteps
        rk4_step!(u, v, t, dt, N, h, pre, ws)
        t += dt
        (all(isfinite, u) && all(isfinite, v)) || error("blow-up at t=$t")
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
