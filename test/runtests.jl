using WesterveltEDG
using Test, Random

# element-by-element RHS of v0.1.0 with the dense LU mass solve (reference)
include("reference_rhs.jl")

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
            a = edg_rhs_reference(u, v, 0.037, N, 1.0 / N, pre)
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
            worst = max(worst, rhs_rel(edg_rhs_reference(u, v, 0.037, N, 1.0 / N, reference_pre(pre)),
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
            worst = max(worst, rhs_rel(edg_rhs_reference(u, v, 0.037, N, 1.0 / N, reference_pre(pre)),
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
end
