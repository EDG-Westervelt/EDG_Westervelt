using WesterveltEDG
using Test

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

    # Regression against the stored convergence data (data/convergence/results.csv).
    @testset "regression" begin
        e = solve(3, 6, 0.1; flux = :upwind, mixed = true)
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
end
