using WesterveltEDG
using Test, Random
using LinearAlgebra: cholesky, Symmetric, BLAS

# element-by-element RHS of v0.1.0 with the dense LU mass solve (reference)
include("reference_rhs.jl")

# The reference reads the dense volume operators and the Cholesky factors of the
# pinned u-stiffness of v0.1.0 from `pre`; they are rebuilt from the coefficient
# values at the points (pre.Cw = c²W, pre.Aw = αW).
function dense_volume(pre)
    E = size(pre.Cw, 2);  N = isqrt(E)
    Pxu = kron(pre.P_u, pre.S_u);  Peu = kron(pre.S_u, pre.P_u)
    Pxv = kron(pre.P_v, pre.S_v);  Pev = kron(pre.S_v, pre.P_v)
    cw(j, k) = vec(pre.Cw[:, j + (k - 1) * N, :]);  aw(j, k) = vec(pre.Aw[:, j + (k - 1) * N, :])
    Ku(j, k) = (K = Pxu' * (cw(j, k) .* Pxu) .+ Peu' * (cw(j, k) .* Peu);  K[1, 1] += 1.0;  K)
    merge(pre, (Mc2_uv = [Pxv' * (cw(j, k) .* Pxu) .+ Pev' * (cw(j, k) .* Peu) for j in 1:N, k in 1:N],
                Ma_vv  = [Pxv' * (aw(j, k) .* Pxv) .+ Pev' * (aw(j, k) .* Pev) for j in 1:N, k in 1:N],
                R_ut   = [cholesky(Symmetric(Ku(j, k))).U for j in 1:N, k in 1:N]))
end

# ---- piecewise coefficients ----
# Callable structs, so that the tests below compile `precompute` once for all
# variants. Step: `lo` where the coordinate `dir` (1 = x, 2 = y) is below `s`,
# `hi` above, `on` on the line itself and `edge` on the boundary line where
# that coordinate is 0. Lin: u₀ = a|x - ½| + b x + y/2 (d = 0) and its
# derivatives ∂_x (d = 1), ∂_y (d = 2), ∂_xx = ∂_yy = 0 (d = 3).
struct Step
    dir::Int;  s::Float64;  lo::Float64;  hi::Float64;  on::Float64;  edge::Float64
end
Step(dir, s, lo, hi; on = hi, edge = lo) = Step(dir, s, lo, hi, on, edge)
(f::Step)(x, y) = (t = f.dir == 1 ? x : y; t == 0 ? f.edge : t < f.s ? f.lo : t == f.s ? f.on : f.hi)
struct Lin
    a::Float64;  b::Float64;  d::Int
end
(f::Lin)(x, y) = f.d == 0 ? f.a * abs(x - 0.5) + f.b * x + 0.5y :
                 f.d == 1 ? (x < 0.5 ? -f.a : f.a) + f.b : f.d == 2 ? 0.5 : 0.0
lin_profile(a, b) = (u0 = Lin(a, b, 0), u0_x = Lin(a, b, 1), u0_y = Lin(a, b, 2), u0_xx = Lin(a, b, 3), u0_yy = Lin(a, b, 3))
# Spike: 1 | 1.5 across x = ¼, continuous across x = ½ but with the value g on
# that line itself.
struct Spike
    g::Float64
end
(f::Spike)(x, y) = x == 0.5 ? f.g : x < 0.25 ? 1.0 : 1.5

# Values a[1..4] on the quadrants x < xs or ≥ xs, y < ys or ≥ ys (so that the
# point value on an interface is that of the upper or right piece), with the
# derivatives zero; `smooth` multiplies a piece by 1 + 0.1 cos(πx) cos(πy).
_z0(x, y) = 0.0
quadrant(xs, ys, a) = (x, y) -> a[1 + (x >= xs) + 2 * (y >= ys)]
function quadrant_coefficients(xs, ys; c = (1.0, 0.6, 0.8, 0.45), alpha = (0.01, 0.004, 0.02, 0.001),
                               beta = (0.0, 0.03, 0.01, 0.02), smooth = false)
    cq = quadrant(xs, ys, c)
    if smooth
        Coefficients((x, y) -> cq(x, y) * (1 + 0.1 * cos(π * x) * cos(π * y)),
                     (x, y) -> -0.1π * cq(x, y) * sin(π * x) * cos(π * y),
                     (x, y) -> -0.1π * cq(x, y) * cos(π * x) * sin(π * y),
                     quadrant(xs, ys, alpha), _z0, _z0, quadrant(xs, ys, beta))
    else
        Coefficients(cq, _z0, _z0, quadrant(xs, ys, alpha), _z0, _z0, quadrant(xs, ys, beta))
    end
end

# Terms of the semi-discrete energy identity for a state (u, v) and its RHS
# (ut, vt) with four walls (κ = 0) and no source:
#     dE/dt = Σ∫ c²∇u_h·∇u̇_h + Σ∫ (1 - 2βv_h) v_h v̇_h = -Σ∫ α|∇v_h|² - D_I,
#     D_I = Σ_F ∫_F (½(c²₁ + c²₂)λ_u + ½(α₁ + α₂)λ_v) [[v_h]]² + τ ([[c²∇u_h]] + [[α∇v_h]])²
# (eq. (2.10) of the paper with the penalty weight of the code), with the
# quadrature rules of the scheme. For piecewise-constant coefficients the
# one-sided values of an element are the values at its centre.
function energy_terms(u, v, ut, vt, pre, coef, N)
    h = 1.0 / N;  nu = pre.qu1;  nv = pre.qv1;  qh = pre.q_high
    z, w = GaussQCofs(nu);  zh, wh = GaussQCofs(qh);  W = w * w';  Wh = wh * wh'
    Pu = pre.P_u;  Su = pre.S_u;  Pv = pre.P_v;  Sv = pre.S_v;  Ph = pre.P_h_v
    crd(i, r) = ((i - 1) * h + i * h .+ h .* r) ./ 2
    T1 = 0.0;  T2 = 0.0;  Dv = 0.0;  scale = 0.0
    for k in 1:N, j in 1:N
        X = repeat(crd(k, z)', nu, 1);  Y = repeat(crd(j, z), 1, nu)
        Xh = repeat(crd(k, zh)', qh, 1);  Yh = repeat(crd(j, zh), 1, qh)
        U = u[j, k, :, :];  Ud = ut[j, k, :, :];  V = v[j, k, :, :];  Vd = vt[j, k, :, :]
        c2 = coef.c.(X, Y) .^ 2;  al = coef.alpha.(X, Y);  bh = coef.beta.(Xh, Yh)
        vq = Ph * V * Ph'
        t1 = W .* c2 .* ((Pu * U * Su') .* (Pu * Ud * Su') .+ (Su * U * Pu') .* (Su * Ud * Pu'))
        t2 = (h^2 / 4) .* Wh .* (1 .- 2 .* bh .* vq) .* vq .* (Ph * Vd * Ph')
        T1 += sum(t1);  T2 += sum(t2);  scale += sum(abs, t1) + sum(abs, t2)
        Dv += sum(W .* al .* ((Pv * V * Sv') .^ 2 .+ (Sv * V * Pv') .^ 2))
    end
    E(n, x) = [JacobiP(x, 0, 0, d - 1) for d in 1:n]
    E1(n, x) = [GradJacobiP(x, 0, 0, d - 1) for d in 1:n]
    DI = 0.0;  wf = (h / 2) .* w
    for k in 1:N, j in 1:N, hz in (true, false)
        j2 = hz ? j + 1 : j;  k2 = hz ? k : k + 1          # element 1 = (j, k), element 2 above / right
        (j2 > N || k2 > N) && continue
        tr(Z, P, e) = hz ? P * (Z' * e) : P * (Z * e)
        U1 = u[j, k, :, :];  V1 = v[j, k, :, :];  U2 = u[j2, k2, :, :];  V2 = v[j2, k2, :, :]
        jv  = tr(V1, Pv, E(nv, 1.0)) .- tr(V2, Pv, E(nv, -1.0))
        du1 = (2 / h) .* tr(U1, Pu, E1(nu, 1.0));  du2 = (2 / h) .* tr(U2, Pu, E1(nu, -1.0))
        dv1 = (2 / h) .* tr(V1, Pv, E1(nv, 1.0));  dv2 = (2 / h) .* tr(V2, Pv, E1(nv, -1.0))
        x1 = crd(k, 0.0);  y1 = crd(j, 0.0);  x2 = crd(k2, 0.0);  y2 = crd(j2, 0.0)
        a1 = coef.c(x1, y1)^2;  a2 = coef.c(x2, y2)^2;  b1 = coef.alpha(x1, y1);  b2 = coef.alpha(x2, y2)
        J = a1 .* du1 .- a2 .* du2 .+ b1 .* dv1 .- b2 .* dv2
        pen = 0.5 * (a1 + a2) * pre.lambda_u + 0.5 * (b1 + b2) * pre.lambda_v
        DI += sum(wf .* (pen .* jv .^ 2 .+ pre.tau .* J .^ 2))
    end
    return (dEdt = T1 + T2, Dv = Dv, DI = DI, scale = scale + Dv + DI)
end

# modal coefficients of the L² projection of f(x, y) on the u-space (exact for
# polynomials of degree ≤ q_u on every element), or with P = pre.P_v on the
# v-space
function project_u(f, pre, N; P = pre.P_u)
    h = 1.0 / N;  nu = pre.qu1;  z, w = GaussQCofs(nu);  n = size(P, 2);  U = zeros(N, N, n, n)
    crd(i) = ((i - 1) * h + i * h .+ h .* z) ./ 2
    for k in 1:N, j in 1:N
        U[j, k, :, :] = (P' .* w') * f.(crd(k)', crd(j)) * (w .* P)
    end
    return U
end

# Standing mode of the two-layer medium c = 1 (s < ½), ½ (s > ½), s = x or y,
# α = β = 0, reflecting walls: u = cos(ωt) Φ(s) with Φ = cos(ωs) (s < ½),
# A cos(2ω(1 - s)) (s > ½), ω = 2 atan √2, A = -√3, so that Φ and c²Φ' are
# continuous at s = ½ while Φ' jumps. Energy-norm error at T of the
# :initial_value run with u(0) = Φ.
function two_layer_error(q, N, T, flux, dir)
    om = 2atan(sqrt(2.0));  A = -sqrt(3.0)
    Φ(s)   = s < 0.5 ? cos(om * s) : A * cos(2om * (1 - s))
    dΦ(s)  = s < 0.5 ? -om * sin(om * s) : 2om * A * sin(2om * (1 - s))
    d2Φ(s) = s < 0.5 ? -om^2 * cos(om * s) : -(2om)^2 * A * cos(2om * (1 - s))
    cs(s) = s < 0.5 ? 1.0 : 0.5
    if dir === :x
        coef = Coefficients((x, y) -> cs(x), _z0, _z0, _z0, _z0, _z0, _z0)
        ip = (u0 = (x, y) -> Φ(x), u0_x = (x, y) -> dΦ(x), u0_y = _z0, u0_xx = (x, y) -> d2Φ(x), u0_yy = _z0)
    else
        coef = Coefficients((x, y) -> cs(y), _z0, _z0, _z0, _z0, _z0, _z0)
        ip = (u0 = (x, y) -> Φ(y), u0_x = _z0, u0_y = (x, y) -> dΦ(y), u0_xx = _z0, u0_yy = (x, y) -> d2Φ(y))
    end
    st = solve(q, N, T; mode = :initial_value, initial_profile = ip, coefficients = coef,
               boundary = :wall, flux = flux, return_state = true)
    fu = cos(om * T) - 1.0;  fv = -om * sin(om * T)          # ũ = u - Φ and v at T, times Φ
    h = st.h;  z, _ = GaussQCofs(q + 1);  e = 0.0
    for j in 1:N, k in 1:N
        S = dir === :x ? repeat((((k - 1) * h + k * h .+ h .* z) ./ 2)', q + 1, 1) :
                         repeat(((j - 1) * h + j * h .+ h .* z) ./ 2, 1, q + 1)
        U = st.u[j, k, :, :];  V = st.v[j, k, :, :]
        ux = (2 / h) .* (st.P_u * U * st.S_u');  uy = (2 / h) .* (st.S_u * U * st.P_u')
        gx = dir === :x ? fu .* dΦ.(S) : zero(S);  gy = dir === :x ? zero(S) : fu .* dΦ.(S)
        e += sum(st.W2d .* (0.5 .* st.err_c2[j, k, :, :] .* ((gx .- ux) .^ 2 .+ (gy .- uy) .^ 2) .+
                            0.5 .* (fv .* Φ.(S) .- st.P_v * V * st.P_v') .^ 2))
    end
    return h / 2 * sqrt(e)
end

@testset "WesterveltEDG" begin

    @testset "quadrature and basis" begin
        x, w = GaussQCofs(5)
        @test sum(w) ≈ 2
        @test sum(w .* x .^ 8) ≈ 2 / 9
        # orthonormality of the Legendre basis
        G = [sum(w .* JacobiP(x, 0, 0, m) .* JacobiP(x, 0, 0, n)) for m in 0:4, n in 0:4]
        @test G ≈ [m == n ? 1.0 : 0.0 for m in 0:4, n in 0:4] atol = 1e-13
    end

    # For the manufactured solution the zero state (ũ_h, v_h) = (0, 0) solves the
    # nonlinear initialization system: u̇_h(0) = 0 and v̇_h(0) is the L²
    # projection of a⁰ v̇⁰ = -ω² Φ (since v⁰ = 0, a⁰ = 1).
    @testset "initialization ($flux, mixed=$mixed)" for flux in (:upwind, :central, :alternating),
                                                        mixed in (true, false)
        N = 4
        pre, _ = precompute(3, N; flux = flux, mixed = mixed)
        u0 = zeros(N, N, pre.qu1, pre.qu1)
        v0 = zeros(N, N, pre.qv1, pre.qv1)
        ut0, vt0 = edg_rhs(u0, v0, 0.0, N, 1.0 / N, pre)
        @test maximum(abs, ut0) < 1e-12
        M = pre.Phi2d_h_v' * (vec(pre.W2d_h) .* pre.Phi2d_h_v)
        r = 0.0;  s = 0.0
        for j in 1:N, k in 1:N
            b = pre.P_h_v' * (pre.W2d_h .* (-pre.omega^2 .* pre.mms_SP[j, k, :, :])) * pre.P_h_v
            proj = reshape(M \ vec(b), pre.qv1, pre.qv1)
            r = max(r, maximum(abs.(vt0[j, k, :, :] .- proj)));  s = max(s, maximum(abs.(proj)))
        end
        @test r / s < 1e-12
    end

    # The batched RHS (src/rhs.jl) against the element-by-element reference
    # implementation of v0.1.0: same scheme, different order of operations
    # and an iterative (CG, tolerance 1e-14) instead of a direct mass solve.
    @testset "batched RHS = reference RHS ($mode, $boundary)" for mode in (:mms, :initial_value, :source),
                                                                  boundary in (:absorbing, :dirichlet)
        Random.seed!(2)
        worst = 0.0
        for q in (1, 3, 4), N in (1, 2, 4), flux in (:upwind, :central, :alternating), mixed in (true, false)
            pre, _ = precompute(q, N; flux = flux, mixed = mixed, mode = mode, boundary = boundary, sigma = 0.3)
            u = randn(N, N, pre.qu1, pre.qu1);  v = randn(N, N, pre.qv1, pre.qv1)
            a = edg_rhs_reference(u, v, 0.037, N, 1.0 / N, dense_volume(pre))
            b = edg_rhs(u, v, 0.037, N, 1.0 / N, pre)
            worst = max(worst, maximum(abs, a[1] .- b[1]) / maximum(abs, a[1]),
                               maximum(abs, a[2] .- b[2]) / maximum(abs, a[2]))
        end
        @test worst < 1e-12
    end

    # Regression against the stored convergence data (data/convergence/results.csv,
    # v0.3.0, default boundary :absorbing_2beta) and, for boundary = :absorbing,
    # against the data of v0.2.0 (weight √(1-βv)).
    @testset "regression" begin
        e = solve(3, 6, 0.1; flux = :upwind, mixed = true)
        @test e.erru ≈ 5.9847841944e-04 rtol = 1e-9
        @test e.errv ≈ 1.9998237454e-02 rtol = 1e-9
        @test e.errgradu ≈ 2.4571643407e-02 rtol = 1e-9
        @test e.errE ≈ 2.2400357448e-02 rtol = 1e-9
        e = solve(3, 6, 0.1; flux = :upwind, mixed = true, boundary = :absorbing)
        @test e.erru ≈ 5.9847813824e-04 rtol = 1e-9
        @test e.errv ≈ 1.9998249758e-02 rtol = 1e-9
        @test e.errgradu ≈ 2.4571634466e-02 rtol = 1e-9
        @test e.errE ≈ 2.2400354434e-02 rtol = 1e-9
    end

    @testset "convergence" begin
        e1 = solve(2, 4, 0.1; flux = :upwind, mixed = true)
        e2 = solve(2, 8, 0.1; flux = :upwind, mixed = true)
        @test log2(e1.erru / e2.erru) > 2.5     # expected ≈ q_u + 1 = 3
        @test log2(e1.errE / e2.errE) > 1.5     # expected ≈ q_u = 2
    end

    # ---- boundary kinds ----
    rhs_rel(a, b) = max(maximum(abs, a[1] .- b[1]) / maximum(abs, a[1]),
                        maximum(abs, a[2] .- b[2]) / maximum(abs, a[2]))
    # The reference RHS knows only the family with the weight √(1-βv) and reads
    # pre.gamma_abc/eta_abc/sigma: a uniform kind is passed through (γ, η, σ) and
    # the face β scaled by k; the wall is the member γ = 0, η = 1, σ = -κ, k = 0.
    function reference_pre(pre)
        b = pre.bc.bot
        g, e, s, k = b.kind === :wall ? (0.0, 1.0, -b.kappa, 0.0) : (b.gamma_abc, b.eta_abc, b.sigma, b.kb)
        merge(pre, (gamma_abc = g, eta_abc = e, sigma = s,
                    bot_bf = k .* pre.bot_bf, top_bf = k .* pre.top_bf,
                    lft_bf = k .* pre.lft_bf, rgt_bf = k .* pre.rgt_bf))
    end
    MIXED = (bot = :absorbing_2beta, top = :dirichlet, lft = :wall, rgt = :absorbing)
    MIXED2 = (bot = :wall, top = :absorbing, lft = :dirichlet, rgt = :absorbing_2beta)

    @testset "boundary spellings (bitwise)" begin
        e = solve(2, 4, 0.05)
        @test e == solve(2, 4, 0.05; boundary = :absorbing_2beta)
        @test e == solve(2, 4, 0.05; boundary = (rgt = :absorbing_2beta, lft = :absorbing_2beta,
                                                 top = :absorbing_2beta, bot = :absorbing_2beta),
                         kappa = 0.7, phase = (0.0, 0.0))
        @test solve(2, 4, 0.05; boundary = :absorbing) ==
              solve(2, 4, 0.05; boundary = (rgt = :absorbing, lft = :absorbing, top = :absorbing, bot = :absorbing))
        D4 = (bot = :dirichlet, top = :dirichlet, lft = :dirichlet, rgt = :dirichlet)
        @test solve(2, 4, 0.05; boundary = :dirichlet, sigma = 0.3) ==
              solve(2, 4, 0.05; boundary = D4, sigma_dirichlet = 0.3)
        pre, _ = precompute(2, 2; boundary = :dirichlet, sigma = 0.3)
        @test pre.gamma_abc == 1.0 && pre.eta_abc == 0.0 && pre.sigma == 0.3
        pre, _ = precompute(2, 2; boundary = MIXED)
        @test isnan(pre.gamma_abc) && isnan(pre.sigma)
        # per-side parameters come from the keywords (the reference RHS reads them from pre.bc)
        bc = precompute(2, 2; boundary = MIXED, sigma = 0.3, sigma_dirichlet = 0.6, kappa = 0.7)[1].bc
        @test (bc.bot.sigma, bc.bot.kb, bc.rgt.sigma, bc.rgt.kb) == (0.3, 2.0, 0.3, 1.0)
        @test (bc.top.gamma_abc, bc.top.sigma, bc.lft.kind, bc.lft.kappa) == (1.0, 0.6, :wall, 0.7)
        # default boundary of precompute
        @test precompute(2, 2)[1].bc == precompute(2, 2; boundary = :absorbing_2beta)[1].bc
    end

    @testset "batched RHS = reference RHS, boundary kinds ($mode, $boundary)" for mode in (:mms, :initial_value, :source),
                                                                                  boundary in (:absorbing_2beta, :dirichlet, :wall)
        Random.seed!(2)
        worst = 0.0
        for q in (1, 3, 4), N in (1, 2, 4), flux in (:upwind, :central, :alternating), mixed in (true, false)
            pre, _ = precompute(q, N; flux = flux, mixed = mixed, mode = mode, boundary = boundary,
                                sigma = 0.3, sigma_dirichlet = 0.6, kappa = 0.7, phase = (0.4, 0.3))
            u = randn(N, N, pre.qu1, pre.qu1);  v = 0.05 .* randn(N, N, pre.qv1, pre.qv1)
            worst = max(worst, rhs_rel(edg_rhs_reference(u, v, 0.037, N, 1.0 / N, dense_volume(reference_pre(pre))),
                                       edg_rhs(u, v, 0.037, N, 1.0 / N, pre)))
        end
        @test worst < 1e-12
    end

    # λ_u ≠ λ_v so that the penalty weights c² λ_u and α λ_v are told apart
    @testset "batched RHS = reference RHS (λ_u ≠ λ_v, $boundary)" for boundary in (:absorbing_2beta, :wall)
        Random.seed!(3)
        worst = 0.0
        fp = (theta = 1.0, tau = 0.4, lambda_u = 0.7, lambda_v = 0.2)
        for q in (1, 3), N in (2, 4), mixed in (true, false), mode in (:mms, :source)
            pre, _ = precompute(q, N; flux_params = fp, mixed = mixed, mode = mode, boundary = boundary,
                                sigma = 0.3, kappa = 0.7)
            u = randn(N, N, pre.qu1, pre.qu1);  v = 0.05 .* randn(N, N, pre.qv1, pre.qv1)
            worst = max(worst, rhs_rel(edg_rhs_reference(u, v, 0.037, N, 1.0 / N, dense_volume(reference_pre(pre))),
                                       edg_rhs(u, v, 0.037, N, 1.0 / N, pre)))
        end
        @test worst < 1e-12
    end

    @testset "face fluxes: consistency and energy" begin
        Random.seed!(4);  n = 1000
        c = 0.7 .+ 0.5 .* rand(n);  c2 = c .^ 2;  a = 1e-2 .* rand(n);  b = 0.3 .* rand(n)
        v = 2 .* rand(n) .- 1;  gu = randn(n);  gv = randn(n);  z = zeros(n)
        psi = c2 .* gu .+ a .* gv;  scale = maximum(psi .^ 2 .+ v .^ 2)
        for kind in (:absorbing, :absorbing_2beta, :dirichlet, :wall), s in (-1.0, 0.0, 0.5, 1.0), kap in (0.0, 1.0)
            bs = WesterveltEDG._boundary_side(kind, s, abs(s), kap)
            # data equal to the traces: no correction
            F1, dv = WesterveltEDG._side_face(bs, c2, a, c, b, v, gu, gv, gu, gv, v)
            @test F1 == psi && all(iszero, dv)
            # homogeneous data: B = ψ (v* - v_h) + v_h F1 against its closed form
            F1, dv = WesterveltEDG._side_face(bs, c2, a, c, b, v, gu, gv, z, z, z)
            B = psi .* dv .+ v .* F1
            if kind === :wall
                Bx = -(kap ./ c) .* psi .^ 2
            else
                g = bs.gamma_abc;  e = bs.eta_abc;  sg = bs.sigma;  W = sqrt.(1 .- bs.kb .* b .* v)
                Bx = (-e * (g - sg * e) .* psi .^ 2 .- g * (e + sg * g) .* (c .* W .* v) .^ 2) ./ (c .* W)
            end
            @test maximum(abs, B .- Bx) <= 1e-12 * scale
            @test maximum(B) <= 1e-12 * scale
        end
    end

    @testset "boundary sides are local" begin
        N = 4
        A4 = (bot = :absorbing, top = :absorbing, lft = :absorbing, rgt = :absorbing)
        S  = (bot = :absorbing_2beta, top = :dirichlet, lft = :wall, rgt = :wall)
        prep(bd) = precompute(3, N; boundary = bd, sigma = 0.3, kappa = 0.5, phase = (0.4, 0.3))[1]
        p0 = prep(A4)
        Random.seed!(9);  u = randn(N, N, p0.qu1, p0.qu1);  v = 0.05 .* randn(N, N, p0.qv1, p0.qv1)
        r(bd) = edg_rhs(u, v, 0.037, N, 1.0 / N, prep(bd))
        r0 = r(A4);  su = copy(r0[1]);  sv = copy(r0[2])
        for (s, adj) in ((:bot, (j, k) -> j == 1), (:top, (j, k) -> j == N),
                         (:lft, (j, k) -> k == 1), (:rgt, (j, k) -> k == N))
            r1 = r(merge(A4, NamedTuple{(s,)}((S[s],))))
            d = [maximum(abs, r1[1][j, k, :, :] .- r0[1][j, k, :, :]) +
                 maximum(abs, r1[2][j, k, :, :] .- r0[2][j, k, :, :]) for j in 1:N, k in 1:N]
            @test all(d[j, k] > 0 for j in 1:N, k in 1:N if adj(j, k))
            @test all(d[j, k] == 0 for j in 1:N, k in 1:N if !adj(j, k))
            su .+= r1[1] .- r0[1];  sv .+= r1[2] .- r0[2]
        end
        @test rhs_rel(r(S), (su, sv)) < 1e-12
    end

    @testset "initialization ($boundary, kappa=$kappa)" for boundary in (:absorbing_2beta, :dirichlet, :wall, MIXED),
                                                           kappa in (0.0, 0.5)
        N = 4
        for flux in (:upwind, :central, :alternating), mixed in (true, false)
            pre, _ = precompute(3, N; flux = flux, mixed = mixed, boundary = boundary, sigma = 0.3,
                                kappa = kappa, phase = (0.4, 0.3))
            ut0, vt0 = edg_rhs(zeros(N, N, pre.qu1, pre.qu1), zeros(N, N, pre.qv1, pre.qv1), 0.0, N, 1.0 / N, pre)
            @test maximum(abs, ut0) < 1e-12
            M = pre.Phi2d_h_v' * (vec(pre.W2d_h) .* pre.Phi2d_h_v)
            r = 0.0;  s = 0.0
            for j in 1:N, k in 1:N
                b = pre.P_h_v' * (pre.W2d_h .* (-pre.omega^2 .* pre.mms_SP[j, k, :, :])) * pre.P_h_v
                proj = reshape(M \ vec(b), pre.qv1, pre.qv1)
                r = max(r, maximum(abs.(vt0[j, k, :, :] .- proj)));  s = max(s, maximum(abs.(proj)))
            end
            @test r / s < 1e-12
        end
    end

    @testset "boundary argument checks" begin
        @test_throws ErrorException precompute(2, 2; boundary = :dirichlet, sigma = -0.5)
        @test_throws ErrorException precompute(2, 2; boundary = :dirichlet, sigma_dirichlet = -0.1)
        @test_throws ErrorException precompute(2, 2; boundary = :absorbing, sigma = 1.5)
        @test_throws ErrorException precompute(2, 2; boundary = :wall, kappa = -1.0)
        @test_throws ErrorException precompute(2, 2; boundary = :neumann)
        @test_throws ErrorException precompute(2, 2; boundary = (bot = :wall, top = :wall, lft = :wall))
        @test_throws ErrorException precompute(2, 2; boundary = (bot = :wall, top = :wall, lft = :wall, rgt = :foo))
        # 1 - 2βv ≤ 0 at the face points of a √(1-2βv) side while 1 - βv > 0 there:
        # pointwise, and in the RHS (where the face check comes before the mass solve)
        n = 5;  o = ones(n);  z = zeros(n);  vb = fill(0.75, n)
        for kind in (:absorbing_2beta, :absorbing)
            bs = WesterveltEDG._boundary_side(kind, 0.0, 0.0, 0.0)
            face(vh, g) = WesterveltEDG._side_face(bs, o, z, o, o, vh, z, z, z, z, g)
            if kind === :absorbing_2beta
                @test_throws ErrorException face(vb, z)
                @test_throws ErrorException face(z, vb)
            else
                @test all(isfinite, face(vb, vb)[2])
            end
        end
        pre, _ = precompute(2, 2; boundary = :absorbing_2beta,
                            coefficients = constant_coefficients(c = 1.0, alpha = 1e-6, beta = 1.0))
        v = zeros(2, 2, pre.qv1, pre.qv1);  v[:, :, 1, 1] .= 1.5
        err = try edg_rhs(zeros(2, 2, pre.qu1, pre.qu1), v, 0.0, 2, 0.5, pre); nothing catch e; e end
        @test err isa ErrorException && occursin("face point", err.msg)
    end

    # energy after the source is switched off: conserved for walls with κ = 0 and
    # Dirichlet sides with σ_D = 0 (central flux), dissipated otherwise
    @testset "boundary energy" begin
        src = GaussianSource(x0 = (0.3, 0.6), r0 = 0.1, omega = 8π, Ts = 0.25)
        co  = constant_coefficients(c = 1.0, alpha = 0.0, beta = 0.0)
        energy_after(; kw...) = (H = solve(3, 8, 0.75; mode = :source, coefficients = co, source = src,
                                          record_every = 5, kw...); H.energy[H.t .> 0.25])
        for kw in ((boundary = :wall, flux = :central), (boundary = :dirichlet, flux = :central),
                   (boundary = (bot = :dirichlet, top = :wall, lft = :wall, rgt = :dirichlet), flux = :central))
            E = energy_after(; kw...)
            @test maximum(abs.(E .- E[1])) / E[1] < 1e-6
        end
        for kw in ((boundary = :wall, kappa = 0.5, flux = :central),
                   (boundary = :dirichlet, sigma_dirichlet = 1.0, flux = :central),
                   (boundary = (bot = :absorbing_2beta, top = :wall, lft = :wall, rgt = :wall),))
            E = energy_after(; kw...)
            @test maximum(diff(E)) <= 1e-10 * E[1]
            @test E[end] / E[1] < 0.95
        end
    end

    @testset "convergence ($boundary)" for boundary in (:absorbing_2beta, :dirichlet, :wall, MIXED)
        e1 = solve(2, 6, 0.1; boundary = boundary, sigma_dirichlet = 1.0, kappa = 1.0, phase = (0.4, 0.3))
        e2 = solve(2, 12, 0.1; boundary = boundary, sigma_dirichlet = 1.0, kappa = 1.0, phase = (0.4, 0.3))
        @test log2(e1.erru / e2.erru) > 2.5
        @test log2(e1.errE / e2.errE) > 1.5
    end

    # ---- mass solvers ----
    # Chebyshev and CG against a dense solve of M(v_h) X = R, element by element,
    # for 1 - 2βv_h from close to 1 down to 0.1; then whole runs with both.
    @testset "mass solvers" begin
        rng = MersenneTwister(7)
        for (q, mixed, N) in ((1, true, 3), (3, true, 4), (4, false, 3), (6, true, 2))
            pre, _ = precompute(q, N; mixed = mixed)
            ws = Workspace(pre, N)
            nv = pre.qv1;  B = pre.Phi2d_h_v;  Wv = vec(pre.W2d_h)
            V0 = randn(rng, nv, nv, N, N);  R = randn(rng, nv, nv, N, N)
            g0 = maximum(maximum(abs, 2 .* vec(pre.beta_hE[:, :, j, k]) .* (B * vec(V0[:, :, j, k])))
                         for j in 1:N, k in 1:N)
            for gmax in (1e-3, 0.5, 0.9)
                V = V0 .* (gmax / g0)
                Xref = similar(R)
                for k in 1:N, j in 1:N
                    a = 1 .- 2 .* vec(pre.beta_hE[:, :, j, k]) .* (B * vec(V[:, :, j, k]))
                    M = B' * ((a .* Wv) .* B)
                    Xref[:, :, j, k] = reshape(M \ vec(R[:, :, j, k]), nv, nv)
                end
                Xc = WesterveltEDG._mass_solve_cheb!(similar(R), V, R, pre, ws, N)
                Xg = WesterveltEDG._mass_solve_cg!(similar(R), V, R, pre, ws, N)
                @test maximum(abs, Xc .- Xref) <= 1e-12 * maximum(abs, Xref)
                @test maximum(abs, Xg .- Xref) <= 1e-12 * maximum(abs, Xref)
            end
            # v_h = 0: M = I, one step
            @test WesterveltEDG._mass_solve_cheb!(similar(R), zero(V0), R, pre, ws, N) == R
            # 1 - 2βv_h ≤ 0 somewhere stops the run
            @test_throws ErrorException WesterveltEDG._mass_solve_cheb!(similar(R), V0 .* (1.5 / g0), R, pre, ws, N)
        end
        for kw in ((mixed = true,), (mixed = false, boundary = :absorbing), (mixed = true, boundary = :dirichlet))
            e1 = solve(3, 6, 0.1; kw..., mass_solver = :cg)
            e2 = solve(3, 6, 0.1; kw..., mass_solver = :chebyshev)
            @test e2.erru ≈ e1.erru rtol = 1e-10
            @test e2.errv ≈ e1.errv rtol = 1e-10
            @test e2.errE ≈ e1.errE rtol = 1e-10
        end
        co = constant_coefficients(c = 1.0, alpha = 1e-6, beta = 8.0)
        H1 = solve(3, 8, 0.3; mode = :source, coefficients = co, mass_solver = :cg)
        H2 = solve(3, 8, 0.3; mode = :source, coefficients = co, mass_solver = :chebyshev)
        @test maximum(abs, H2.recv .- H1.recv) <= 1e-10 * maximum(abs, H1.recv)
        @test_throws ErrorException precompute(2, 2; mass_solver = :lu)
    end

    # ---- multithreading ----
    # Every number of blocks gives the same result bit for bit. The blocks run
    # as parallel tasks, on one thread as well; N is odd, so that the blocks
    # differ in size and mesh columns are split between them. The coefficients
    # are the smooth ones of the paper and piecewise ones that jump on the mesh
    # lines x = 3/7 and y = 4/7 (the one-sided face kernel; not in :mms mode,
    # which needs continuous c and α).
    @info "multithreading tests on $(Threads.nthreads()) Julia thread(s)"
    PW7 = quadrant_coefficients(3 / 7, 4 / 7; alpha = (1e-5, 4e-6, 2e-5, 1e-6), smooth = true)
    @testset "blocks: bitwise RHS and RK4 ($solver, q_u = $q)" for solver in (:chebyshev, :cg), q in (1, 2, 3, 6)
        N = 7;  h = 1.0 / N
        nblas = BLAS.get_num_threads();  BLAS.set_num_threads(1)
        try
            for mixed in (true, false), mode in (:mms, :initial_value, :source), boundary in (MIXED, MIXED2),
                coef in (paper_coefficients(), PW7)
                mode === :mms && coef === PW7 && continue
                pre, _ = precompute(q, N; mixed = mixed, mode = mode, boundary = boundary, mass_solver = solver,
                                    sigma = 0.3, sigma_dirichlet = 0.6, kappa = 0.7, phase = (0.4, 0.3),
                                    coefficients = coef)
                @test pre.one_sided == (coef === PW7)
                rng = MersenneTwister(q)
                u = 1e-2 .* randn(rng, N, N, pre.qu1, pre.qu1);  v = 0.05 .* randn(rng, N, N, pre.qv1, pre.qv1)
                ws1 = Workspace(pre, N; nblocks = 1)
                r1 = copy.(edg_rhs!(ws1, u, v, 0.037, N, h, pre))
                s1 = (copy(u), copy(v))
                for k in 1:2
                    rk4_step!(s1..., 0.01k, 1e-4, N, h, pre, ws1)
                end
                for nb in (3, 5, 8)
                    ws = Workspace(pre, N; nblocks = nb)
                    @test ws.nb == nb
                    @test edg_rhs!(ws, u, v, 0.037, N, h, pre) == r1
                    s = (copy(u), copy(v))
                    for k in 1:2
                        rk4_step!(s..., 0.01k, 1e-4, N, h, pre, ws)
                    end
                    @test s == s1
                end
            end
        finally
            BLAS.set_num_threads(nblas)
        end
    end

    # Chebyshev step counts that differ between the blocks: 1 - 2βv_h spreads
    # over [0.45, 1.55] on the first two mesh columns (about 28 steps) and stays
    # close to 1 elsewhere; all elements take the largest count.
    @testset "blocks: Chebyshev step counts" begin
        N = 9;  h = 1.0 / N
        co = replace_coefficients(paper_coefficients(); beta = 8.0)
        pre, _ = precompute(4, N; coefficients = co, mode = :source)
        rng = MersenneTwister(5)
        u = 1e-3 .* randn(rng, N, N, pre.qu1, pre.qu1);  v = 1e-4 .* randn(rng, N, N, pre.qv1, pre.qv1)
        v[:, 1:2, 2, 1] .= 0.04
        r1 = edg_rhs(u, v, 0.1, N, h, pre; nblocks = 1)
        for nb in (2, 3, 5, 8, 20)
            ws = Workspace(pre, N; nblocks = nb)
            @test edg_rhs!(ws, u, v, 0.1, N, h, pre) == r1
            @test maximum(ws.nsteps) > 20 && minimum(ws.nsteps) < 10
        end
    end

    @testset "blocks: solve ($solver, $mode)" for solver in (:chebyshev, :cg), mode in (:mms, :initial_value, :source)
        nblas = BLAS.get_num_threads()
        for (q, mixed) in ((1, false), (2, true), (3, false), (6, true)), coef in (paper_coefficients(), PW7)
            mode === :mms && coef === PW7 && continue
            sol(nb) = solve(q, 7, 0.01; mixed = mixed, mode = mode, mass_solver = solver, nblocks = nb,
                            boundary = MIXED, sigma = 0.3, sigma_dirichlet = 0.6, kappa = 0.7,
                            coefficients = coef, phase = coef === PW7 ? (0.4, 0.3) : (0.0, 0.0),
                            return_state = mode === :initial_value)
            r1 = sol(1)
            for nb in (3, 5, 8)
                @test sol(nb) == r1
            end
        end
        @test BLAS.get_num_threads() == nblas
    end

    # concurrent runs in one session: same results, BLAS threads restored at the end
    @testset "blocks: concurrent solves" begin
        nblas = BLAS.get_num_threads()
        r1 = solve(2, 7, 0.01; nblocks = 1)
        tasks = [Threads.@spawn solve(2, 7, 0.01; nblocks = nb) for nb in (3, 5, 8, 3)]
        @test all(==(r1), fetch.(tasks))
        @test BLAS.get_num_threads() == nblas
    end

    # errors raised inside the parallel regions keep their type and message, and
    # the first one in the order of the serial loops is reported
    @testset "blocks: errors" begin
        N = 8;  h = 1.0 / N
        co = replace_coefficients(paper_coefficients(); alpha = 1e-6, beta = 1.0)
        geterr(f) = try f(); nothing catch e; e end
        bd = (bot = :absorbing_2beta, top = :absorbing, lft = :absorbing, rgt = :wall)
        for solver in (:chebyshev, :cg)
            pre, _ = precompute(3, N; coefficients = co, boundary = bd, mass_solver = solver)
            u = zeros(N, N, pre.qu1, pre.qu1)
            # non-positive weights on the bottom side (last column) and on the left
            # side (row 4), where the mass matrices fail as well
            v = zeros(N, N, pre.qv1, pre.qv1);  v[1, N, 1, 1] = 2.5;  v[4, 1, 1, 1] = 2.5
            # non-positive mass matrices on the interior elements (5, 2) and (4, 6)
            w = zeros(N, N, pre.qv1, pre.qv1);  w[4, 6, 1, 1] = 2.5;  w[5, 2, 1, 1] = 2.5
            mass_msg = solver === :chebyshev ? "mass matrix not positive definite on element 13: 1 - 2βv_h ≤ 0" :
                                               "mass matrix not positive definite on element (5, 2): 1 - 2βv_h ≤ 0"
            for nb in (1, 3, 8)
                e = geterr(() -> edg_rhs(u, v, 0.0, N, h, pre; nblocks = nb))
                @test e isa ErrorException && e.msg == "boundary weight 1 - 2βv_h ≤ 0 at a face point (absorbing_2beta side)"
                e = geterr(() -> edg_rhs(u, w, 0.0, N, h, pre; nblocks = nb))
                @test e isa ErrorException && e.msg == mass_msg
            end
        end
        # boundary data with 1 - βg ≤ 0 (manufactured solution, β = 1), first on the left side
        pre, _ = precompute(3, N; coefficients = co, phase = (0.4, 0.3),
                            boundary = (bot = :wall, top = :wall, lft = :absorbing, rgt = :absorbing_2beta))
        for nb in (1, 3, 8)
            e = geterr(() -> edg_rhs(zeros(N, N, pre.qu1, pre.qu1), zeros(N, N, pre.qv1, pre.qv1), 0.07, N, h, pre;
                                     nblocks = nb))
            @test e isa ErrorException && e.msg == "boundary weight 1 - βg ≤ 0 for the boundary data g (absorbing side)"
        end
        @test_throws ErrorException Workspace(pre, N; nblocks = 0)
        @test Workspace(pre, 3; nblocks = 8).nb == 2
        # other errors inside a task are rethrown as the TaskFailedException
        @test geterr(() -> WesterveltEDG._foreach_block(b -> b >= 2 && throw(DomainError(b)), 4)) isa TaskFailedException
        e = geterr(() -> WesterveltEDG._foreach_block(b -> b >= 2 && error("block $b"), 4))
        @test e isa ErrorException && e.msg == "block 2"
    end

    # state, workspace and pre of different sizes are refused before the kernels
    # (which index with @inbounds); the default number of blocks
    @testset "blocks: sizes and default" begin
        pre, _ = precompute(2, 8);  pre9, _ = precompute(2, 9)
        for nb in (1, 3)
            @test_throws DimensionMismatch edg_rhs(zeros(8, 8, 3, 3), zeros(8, 7, 2, 2), 0.0, 8, 1 / 8, pre; nblocks = nb)
            @test_throws DimensionMismatch edg_rhs!(Workspace(pre9, 9; nblocks = nb), zeros(8, 8, 3, 3), zeros(8, 8, 2, 2),
                                                    0.0, 8, 1 / 8, pre)
        end
        @test_throws DimensionMismatch edg_rhs(zeros(8, 8, 3, 3), zeros(8, 8, 2, 2), 0.0, 8, 1 / 8, pre9)
        @test Workspace(pre, 8).nb == 1                                     # N²(q_u+1)² = 576
        @test Workspace(precompute(4, 16)[1], 16).nb == min(4, Threads.nthreads())    # 6400
        @test Workspace(precompute(4, 32)[1], 32).nb == Threads.nthreads()            # 25600
    end

    # the energy diagnostics of :source mode (plain loops on the blocks) equal
    # the serial expressions with the products of `*`, bit for bit
    @testset "blocks: energy diagnostics" begin
        N = 5;  h = 1.0 / N
        for q in 1:6, mixed in (true, false)
            pre, geo = precompute(q, N; mixed = mixed, mode = :source,
                                  coefficients = replace_coefficients(paper_coefficients(); beta = 6.0))
            rng = MersenneTwister(q)
            u = 1e-2 .* randn(rng, N, N, pre.qu1, pre.qu1);  v = 1e-2 .* randn(rng, N, N, pre.qv1, pre.qv1)
            ws = Workspace(pre, N; nblocks = 3)
            dg = WesterveltEDG._diag_workspace(pre, N, ws)
            WesterveltEDG._energy_diag!(dg, u, v, pre, geo, N, h, ws)
            same = true
            for k in 1:N, j in 1:N
                ux = (2/h) .* (pre.P_u * u[j,k,:,:] * geo.S_u');  uy = (2/h) .* (geo.S_u * u[j,k,:,:] * pre.P_u')
                vh = pre.P_v * v[j,k,:,:] * pre.P_v'
                c2q = geo.err_c2[j,k,:,:];  bq = geo.err_beta[j,k,:,:];  e = j + (k - 1) * N
                same &= dg.E[e] == (h^2/4) * sum(geo.W2d .* (0.5 .* c2q .* (ux.^2 .+ uy.^2)
                                                             .+ (0.5 .- (2/3) .* bq .* vh) .* vh.^2))
                same &= dg.vmax[e] == maximum(abs.(vh)) && dg.ndmin[e] == minimum(1 .- 2 .* bq .* vh)
            end
            @test same
        end
    end

    # ---- initial profile ----
    @testset "initial profile" begin
        Φ = (x, y) -> cos(2π*x) * sin(2π*y)
        prof = (u0 = Φ, u0_x = (x, y) -> -2π * sin(2π*x) * sin(2π*y),
                u0_y = (x, y) -> 2π * cos(2π*x) * cos(2π*y),
                u0_xx = (x, y) -> -(2π)^2 * Φ(x, y), u0_yy = (x, y) -> -(2π)^2 * Φ(x, y))
        co = replace_coefficients(paper_coefficients(); alpha = 1e-6)
        A = solve(3, 6, 0.05; mode = :initial_value, coefficients = co, return_state = true)
        B = solve(3, 6, 0.05; mode = :initial_value, coefficients = co, return_state = true,
                  initial_profile = prof)
        @test A.u == B.u && A.v == B.v
        @test_throws ErrorException precompute(2, 2; initial_profile = prof)
    end

    # ---- piecewise coefficients (jumps on mesh faces) ----
    # Continuous coefficients keep the plain point values on the faces, bit for
    # bit, and the kernel with one value per face point; a jump gives the values
    # of the two pieces whichever piece the point value belongs to.
    @testset "piecewise coefficients: face values" begin
        for (q, N, kw) in ((2, 4, (;)), (3, 5, (; mode = :initial_value, phase = (0.4, 0.3))),
                           (1, 3, (; coefficients = constant_coefficients(c = 1.0, alpha = 0.0, beta = 0.0))))
            pre, _ = precompute(q, N; kw...)
            co = get(kw, :coefficients, paper_coefficients())
            h = 1.0 / N;  z, _ = GaussQCofs(q + 1);  crd(i) = ((i - 1) * h + i * h .+ h .* z) ./ 2
            ok = true
            for k in 1:N, jf in 1:N+1
                xf = crd(k);  yf = (jf - 1) * h
                ok &= pre.hc2[:, jf, k] == co.c.(xf, yf) .^ 2 && pre.haf[:, jf, k] == co.alpha.(xf, yf) &&
                      pre.hcf[:, jf, k] == co.c.(xf, yf) && pre.hbf[:, jf, k] == co.beta.(xf, yf)
                xf, yf = yf, crd(k)                         # vertical faces: column jf, row k
                ok &= pre.vc2[:, k, jf] == co.c.(xf, yf) .^ 2 && pre.vaf[:, k, jf] == co.alpha.(xf, yf) &&
                      pre.vcf[:, k, jf] == co.c.(xf, yf) && pre.vbf[:, k, jf] == co.beta.(xf, yf)
            end
            @test ok
            @test pre.hc2m == pre.hc2 && pre.vc2m == pre.vc2 && pre.hafm == pre.haf && pre.vafm == pre.vaf
            @test all(iszero, pre.hJC) && all(iszero, pre.vJC)
            @test !pre.one_sided
        end
        # c = 1 | 2 across the line x = s (face column kf), defined in different
        # ways: the "-" side sees c² = 1, the "+" side 4; other faces are untouched
        const_step(a) = Step(1, 2.0, a, a)
        step_coef(c; beta = const_step(0.0)) = Coefficients(c, _z0, _z0, const_step(1e-3), _z0, _z0, beta)
        for (N, kf, cfun) in ((4, 3, Step(1, 0.5, 1.0, 2.0)),                  # x < ½ ? 1 : 2
                              (4, 3, Step(1, 0.5, 1.0, 2.0; on = 1.0)),        # x ≤ ½ ? 1 : 2
                              (4, 3, Step(1, 0.5, 1.0, 2.0; on = 7.0)),        # removable value on the line
                              (10, 4, Step(1, 0.3, 1.0, 2.0)),                 # x < 0.3, mesh line 3·0.1 > 0.3
                              (10, 4, Step(1, 0.3, 1.0, 2.0; on = 1.0)))
            pre, _ = precompute(2, N; mode = :source, coefficients = step_coef(cfun))
            rest = setdiff(1:N+1, kf)
            @test all(==(1.0), pre.vc2m[:, :, kf]) && all(==(4.0), pre.vc2[:, :, kf])
            @test pre.vc2m[:, :, rest] == pre.vc2[:, :, rest] && pre.hc2m == pre.hc2 && pre.one_sided
        end
        # boundary values seen from inside: c given for 0 < x (2 on x = 0), β for
        # y > 0 (5 on y = 0)
        pre, _ = precompute(2, 4; mode = :source,
                            coefficients = step_coef(Step(1, 0.5, 1.0, 2.0; edge = 2.0); beta = Step(2, 2.0, 0.1, 0.1; edge = 5.0)))
        @test all(==(1.0), pre.vc2[:, :, 1]) && all(==(1.0), pre.vcf[:, :, 1]) && all(==(0.1), pre.hbf[:, 1, :])
        # a value far off on an interface line (huge, Inf, NaN) hides no jump on
        # another line: c = 1 | 1.5 across x = ¼ (face column 3 for N = 8), and
        # c = 1.5 on both sides of x = ½ (face column 5)
        for g in (1e8, Inf, NaN)
            pre, _ = precompute(2, 8; mode = :source, coefficients = step_coef(Spike(g)))
            @test all(==(1.0), pre.vc2m[:, :, 3]) && all(==(2.25), pre.vc2[:, :, 3])
            @test all(==(2.25), pre.vc2m[:, :, 5]) && all(==(2.25), pre.vc2[:, :, 5])
            @test all(isfinite, pre.vc2m) && all(isfinite, pre.vc2) && all(isfinite, pre.vcf) && pre.one_sided
        end
        # jumps of α alone; of β alone (β only enters the volume and the boundary)
        pre, _ = precompute(2, 4; mode = :source, coefficients = quadrant_coefficients(0.5, 0.5; c = (1.0, 1.0, 1.0, 1.0)))
        @test pre.one_sided && pre.hc2m == pre.hc2 && pre.hafm != pre.haf
        pre, _ = precompute(2, 4; mode = :source,
                            coefficients = quadrant_coefficients(0.5, 0.5; c = (1.0, 1.0, 1.0, 1.0), alpha = (1e-3, 1e-3, 1e-3, 1e-3)))
        @test !pre.one_sided
    end

    # the one-sided kernel reduces to the kernel with one value per face point
    # for continuous coefficients (up to the order of the operations)
    @testset "piecewise coefficients: kernel reduction" begin
        Random.seed!(5)
        for flux in (:upwind, :central), mode in (:mms, :initial_value)
            fp = flux === :upwind ? (theta = 0.3, tau = 0.7, lambda_u = 0.4, lambda_v = 0.9) : flux_parameters(flux)
            pre, _ = precompute(3, 4; flux_params = fp, mode = mode, boundary = MIXED, phase = (0.4, 0.3))
            u = randn(4, 4, pre.qu1, pre.qu1);  v = 0.05 .* randn(4, 4, pre.qv1, pre.qv1)
            @test rhs_rel(edg_rhs(u, v, 0.037, 4, 0.25, pre), edg_rhs(u, v, 0.037, 4, 0.25, merge(pre, (one_sided = true,)))) < 1e-14
        end
    end

    # The batched RHS against the element-by-element reference, which builds the
    # one-sided fluxes from the side values of the two elements, for c, α, β
    # jumping across x = ½ and y = ½ (c also varying inside the pieces), in the
    # modes that allow jumps (with the interface terms of u₀)
    @testset "piecewise coefficients: batched RHS = reference RHS ($mode)" for mode in (:initial_value, :source)
        Random.seed!(6)
        worst = 0.0;  os = true
        co = quadrant_coefficients(0.5, 0.5; smooth = true)
        for q in (1, 3), N in (2, 4), mixed in (true, false), boundary in (:absorbing_2beta, :dirichlet, :wall),
            fp in (flux_parameters(:upwind), flux_parameters(:central), flux_parameters(:alternating),
                   (theta = 0.3, tau = 0.7, lambda_u = 0.4, lambda_v = 0.9))
            pre, _ = precompute(q, N; flux_params = fp, mixed = mixed, mode = mode, boundary = boundary, coefficients = co,
                                sigma = 0.3, sigma_dirichlet = 0.6, kappa = 0.7, phase = (0.4, 0.3))
            os &= pre.one_sided
            u = randn(N, N, pre.qu1, pre.qu1);  v = 0.05 .* randn(N, N, pre.qv1, pre.qv1)
            worst = max(worst, rhs_rel(edg_rhs_reference(u, v, 0.037, N, 1.0 / N, dense_volume(reference_pre(pre))),
                                       edg_rhs(u, v, 0.037, N, 1.0 / N, pre)))
        end
        @test os
        @test worst < 1e-12
    end

    # Semi-discrete energy identity for random states, four walls (κ = 0), no
    # source, c, α and β jumping on the quadrant lines x = ½, y = ½:
    # dE/dt = -Σ∫α|∇v_h|² exactly for central and alternating fluxes (no face
    # dissipation), and dE/dt = -Σ∫α|∇v_h|² - D_I ≤ 0 with the one-sided D_I
    # otherwise.
    @testset "piecewise coefficients: energy identity" begin
        co = quadrant_coefficients(0.5, 0.5; beta = (0.0, 0.3, 0.1, 0.2))
        N = 4;  worst = 0.0;  rise = -Inf
        for q in 1:4, mixed in (true, false),
            fp in (flux_parameters(:central), flux_parameters(:alternating),
                   (theta = 0.3, tau = 0.0, lambda_u = 0.0, lambda_v = 0.0), flux_parameters(:upwind),
                   (theta = 0.5, tau = 1.0, lambda_u = 0.0, lambda_v = 0.0), (theta = 0.3, tau = 0.7, lambda_u = 0.4, lambda_v = 0.9))
            pre, _ = precompute(q, N; flux_params = fp, mixed = mixed, mode = :source, boundary = :wall, kappa = 0.0,
                                coefficients = co, source = GaussianSource(amplitude = 0.0))
            rng = MersenneTwister(10q + mixed)
            u = randn(rng, N, N, pre.qu1, pre.qu1);  v = 0.05 .* randn(rng, N, N, pre.qv1, pre.qv1)
            ut, vt = edg_rhs(u, v, 0.3, N, 1.0 / N, pre)
            et = energy_terms(u, v, ut, vt, pre, co, N)
            fp.tau == 0 && fp.lambda_u == 0 && fp.lambda_v == 0 && @test et.DI == 0
            worst = max(worst, abs(et.dEdt + et.Dv + et.DI) / et.scale)
            rise = max(rise, et.dEdt / et.scale)
        end
        @test worst < 1e-12
        @test rise < 1e-12
    end

    # ũ = u - u₀ at an interface: for u₀ in U_h, the :initial_value RHS at
    # (ũ, v) = (0, 0) equals the plain RHS at (u, v) = (u₀, 0); with c jumping
    # where ∂_n u₀ ≠ 0, and with u₀ = |x - ½| for a constant c
    @testset "piecewise coefficients: interface terms of u₀" begin
        N = 8;  h = 1.0 / N
        # c = 1 | ½ at x = ½ with u₀ = x + y/2; c = 0.8 with u₀ = |x - ½| + y/2
        for (c1, c2, ip) in ((1.0, 0.5, lin_profile(0.0, 1.0)), (0.8, 0.8, lin_profile(1.0, 0.0)))
            u0 = ip.u0
            coef = Coefficients(Step(1, 0.5, c1, c2), _z0, _z0, Step(1, 0.5, 1e-3 * c1^2, 1e-3 * c2^2), _z0, _z0, _z0)
            for flux in (:upwind, :central, :alternating), q in (1, 3)
                kw = (coefficients = coef, boundary = :wall, flux = flux)
                pI, _ = precompute(q, N; mode = :initial_value, initial_profile = ip, kw...)
                pS, _ = precompute(q, N; mode = :source, source = GaussianSource(amplitude = 0.0), kw...)
                @test pI.one_sided
                rI = edg_rhs(zeros(N, N, pI.qu1, pI.qu1), zeros(N, N, pI.qv1, pI.qv1), 0.3, N, h, pI)
                rS = edg_rhs(project_u(u0, pS, N), zeros(N, N, pS.qv1, pS.qv1), 0.3, N, h, pS)
                @test maximum(abs, rI[2] .- rS[2]) <= 1e-11 * maximum(abs, rS[2])
                @test maximum(abs, rI[1] .- rS[1]) <= 1e-11
            end
        end
    end

    # Exact consistency at an interface, without a reference implementation:
    # c and α jump across y = ½ (or x = ½), four walls, β = 0, no source. The
    # states u = w_u(s), v = w_v(s), s = y (or x), with the piecewise-quadratic
    # profiles w = s² (s < ½), A + B(1 - s)² (s > ½) that satisfy [[w]] = 0 and
    # [[κ ∂_s w]] = 0 at s = ½ (κ = c² for w_u, α for w_v) and ∂_s w = 0 on the
    # walls, lie in U_h × V_h and solve the transmission problem, so the scheme
    # reproduces the exact u_t = v, v_t = ∇·(c²∇u) + ∇·(α∇v) to round-off
    # (with one value of c² and α per face point the errors are O(1)).
    @testset "piecewise coefficients: exact consistency at an interface" begin
        c1, c2 = 1.0, 0.6;  a1, a2 = 0.02, 0.005
        # w and (κ w')' for κ = k1 (s < ½), k2 (s > ½): B = -k1/k2, A = (1 - B)/4
        prof(k1, k2) = (B = -k1 / k2;  A = 0.25 * (1 - B);
                        (s -> s < 0.5 ? s^2 : A + B * (1 - s)^2, s -> s < 0.5 ? 2k1 : 2k2 * B))
        wu, du = prof(c1^2, c2^2);  wv, dv = prof(a1, a2)
        worst = 0.0
        for dir in (1, 2), (q, N, mixed) in ((2, 4, false), (3, 4, true), (3, 6, false), (4, 8, true)),
            fp in (flux_parameters(:upwind), flux_parameters(:central), flux_parameters(:alternating),
                   (theta = 0.0, tau = 0.0, lambda_u = 0.0, lambda_v = 0.0), (theta = 0.3, tau = 0.8, lambda_u = 0.5, lambda_v = 2.0))
            co = Coefficients(Step(dir, 0.5, c1, c2), _z0, _z0, Step(dir, 0.5, a1, a2), _z0, _z0, _z0)
            pre, _ = precompute(q, N; mixed = mixed, flux_params = fp, boundary = :wall, kappa = 0.7, coefficients = co,
                                mode = :source, source = GaussianSource(amplitude = 0.0))
            sel(x, y) = dir == 1 ? x : y
            U = project_u((x, y) -> wu(sel(x, y)), pre, N);  V = project_u((x, y) -> wv(sel(x, y)), pre, N; P = pre.P_v)
            Ue = project_u((x, y) -> wv(sel(x, y)), pre, N)
            Ve = project_u((x, y) -> du(sel(x, y)) + dv(sel(x, y)), pre, N; P = pre.P_v)
            ut, vt = edg_rhs(U, V, 0.3, N, 1.0 / N, pre)
            worst = max(worst, maximum(abs, ut .- Ue) / maximum(abs, Ue), maximum(abs, vt .- Ve) / maximum(abs, Ve))
        end
        @test worst < 1e-9
    end

    # Convergence at a coefficient jump: the exact standing mode of a two-layer
    # medium (energy-norm rate q_u for upwind, about 1 for central fluxes at
    # q_u = 2) and the same for an interface along y. The manufactured
    # solution needs continuous c and α: a jump of c or α is refused in :mms
    # mode, a jump of β alone is accepted.
    @testset "piecewise coefficients: convergence" begin
        e8 = two_layer_error(2, 8, 0.5, :upwind, :x);  e16 = two_layer_error(2, 16, 0.5, :upwind, :x)
        @test log2(e8 / e16) > 1.8
        @test two_layer_error(2, 16, 0.5, :upwind, :y) ≈ e16 rtol = 1e-10
        c8 = two_layer_error(2, 8, 0.5, :central, :x);  c16 = two_layer_error(2, 16, 0.5, :central, :x)
        @test log2(c8 / c16) > 0.9
        co = quadrant_coefficients(0.5, 0.5; c = (1.0, 0.7, 0.8, 0.6), alpha = (1e-3, 1e-3, 2e-3, 2e-3),
                                   beta = (1e-4, 0.0, 1e-4, 0.0), smooth = true)
        @test_throws ErrorException precompute(2, 6; coefficients = co)
        @test_throws ErrorException precompute(2, 6; coefficients = quadrant_coefficients(0.5, 0.5;
                                        c = (1.0, 1.0, 1.0, 1.0), alpha = (1e-3, 1e-3, 2e-3, 2e-3)))
        cb = quadrant_coefficients(0.5, 0.5; c = (1.0, 1.0, 1.0, 1.0), alpha = (1e-3, 1e-3, 1e-3, 1e-3))
        @test !precompute(2, 6; coefficients = cb)[1].one_sided
        e1 = solve(2, 6, 0.1; coefficients = cb, boundary = MIXED, kappa = 0.5, sigma_dirichlet = 1.0, phase = (0.4, 0.3))
        e2 = solve(2, 12, 0.1; coefficients = cb, boundary = MIXED, kappa = 0.5, sigma_dirichlet = 1.0, phase = (0.4, 0.3))
        @test log2(e1.erru / e2.erru) > 2.5
        @test log2(e1.errE / e2.errE) > 1.5
    end

    # energy after the source is switched off, c, α, β jumping on x = ½, y = ½,
    # absorbing sides: non-increasing for upwind and central fluxes
    @testset "piecewise coefficients: energy decay" begin
        co = quadrant_coefficients(0.5, 0.5; alpha = (1e-5, 4e-6, 2e-5, 1e-6), beta = (1e-4, 3e-4, 0.0, 2e-4))
        src = GaussianSource(x0 = (0.45, 0.55), r0 = 0.08, amplitude = 50.0, omega = 8π, Ts = 0.25)
        for flux in (:upwind, :central)
            H = solve(3, 8, 0.75; mode = :source, coefficients = co, flux = flux, source = src, record_every = 5)
            E = H.energy[H.t .> 0.25]
            @test maximum(diff(E)) <= 1e-10 * E[1]
            @test E[end] / E[1] < 0.95
        end
    end
end
