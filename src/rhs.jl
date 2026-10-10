# Semi-discrete right-hand side of the EDG scheme for the Westervelt equation
# in first-order-in-time form  u_t = v,
#     (1 - 2βv) v_t = ∇·(c²∇u) + ∇·(α∇v) + f,
# on a uniform Cartesian mesh of the unit square with tensor-product Legendre
# bases of degree q_u (for u) and q_v (for v).
#
# Layout of the state: u[j, k, :, :] holds the (q_u+1)×(q_u+1) modal
# coefficients on the element in row j (y-direction) and column k
# (x-direction); likewise v. On an element the coefficient matrix U[a, b] has
# the y-mode a in the first and the x-mode b in the second index, so that the
# values at the tensor Gauss points are P U P'.
#
# The displacement is advanced as ũ = u - u₀ with u₀ the initial profile
# (see `solve`), which only adds the known terms ∇·(c²∇u₀) to the source and
# -(c²∇u₀)·n to the boundary data of the flux c²∇u·n (absorbing and wall
# sides).
#
# Structure of one evaluation (the element- and face-sized arrays are
# preallocated in a `Workspace`; only small boundary arrays are allocated):
#   1. copy u, v to element-last arrays U[:, :, j, k] (and their transposes);
#   2. traces on the four sides of every element (values and normal
#      derivatives at the face Gauss points) by a few large matrix products;
#   3. numerical fluxes on all faces at once; every interior face is visited
#      once and yields the contributions to both neighbouring elements;
#   4. lift of the face contributions back to the elements;
#   5. volume terms, manufactured source and the u-equation element by element;
#   6. the mass system ∫(1 - 2βv_h) φ v̇_h = R on every element by matrix-free
#      conjugate gradients (see `_mass_solve_cg!`).
# The result is that of the element-by-element version of v0.1.0 (kept in
# test/reference_rhs.jl, dense LU mass solve; the boundary kinds added later
# are compared with it through equivalent parameters) up to the order of
# floating-point operations and the CG tolerance pre.mass_tol = 1e-14 of the
# mass solve.

# Pointwise fluxes on a physical boundary face. Inputs are values at the face
# quadrature points (arrays of any common shape); `*_ex` are the boundary data.
# Every kind returns (F1, dv): the total numerical normal flux
# ((c²∇u)* + (α∇v)*)·n entering the v-equation and the v-trace correction
# v* - v_h. The lift of (F1, dv) in `_boundary_faces!` is the same for all kinds.

# Boundary family ρ = η (c²∇u + α∇v)·n + γ c √(1-kβv) v with η = √(1-γ²)
# (see `_boundary_side` in solver.jl); `kb` = k is 1 for the weight √(1-βv)
# and 2 for √(1-2βv).
@inline function _abc_face(gamma_abc, eta_abc, sigma, kb, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
    rho_num = eta_abc .* (c2f .* grad_u_n .+ af .* grad_v_n) .+
              gamma_abc .* cf .* sqrt.(1.0 .- kb .* bf .* v_fp) .* v_fp
    rho_ex  = eta_abc .* (c2f .* gu_ex .+ af .* gv_ex) .+
              gamma_abc .* cf .* sqrt.(1.0 .- kb .* bf .* v_ex) .* v_ex
    rho_net = rho_num .- rho_ex

    sqf = sqrt.(1.0 .- kb .* bf .* v_fp)
    F1  = c2f .* grad_u_n .+ af .* grad_v_n .- (eta_abc + sigma * gamma_abc) .* rho_net
    dv  = -(gamma_abc - sigma * eta_abc) ./ (cf .* sqf) .* rho_net
    return F1, dv
end

# Reflecting wall (c²∇u + α∇v)·n = g₂ with g₂ = c² gu_ex + α gv_ex (ũ-form data).
# Residual ρ_w = (c²∇u_h + α∇v_h)·n - g₂;  v* = v_h - (κ/c) ρ_w and
# w* = w_h - ½ ρ_w n for w_h ∈ {c²∇u_h, α∇v_h}, so the total numerical normal
# flux is g₂. Face energy contribution -(κ/c) ρ_w² ≤ 0 when g₂ = 0.
@inline function _wall_face(kappa, c2f, af, cf, grad_u_n, grad_v_n, gu_ex, gv_ex)
    g2    = c2f .* gu_ex .+ af .* gv_ex
    rho_w = c2f .* grad_u_n .+ af .* grad_v_n .- g2
    dv    = -(kappa ./ cf) .* rho_w
    return g2, dv
end

# Flux on one side with the parameters `bs` of its boundary kind.
@inline function _side_face(bs, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
    bs.kind === :wall && return _wall_face(bs.kappa, c2f, af, cf, grad_u_n, grad_v_n, gu_ex, gv_ex)
    # the weight needs 1 - kβv > 0 at the face points for the trace v_h and the
    # data (the mass solve only sees the volume points); a NaN passes these
    # comparisons and propagates (in :source mode the blow-up check stops it)
    @inbounds for i in eachindex(bf, v_fp, v_ex)
        1.0 - bs.kb * bf[i] * v_fp[i] <= 0.0 &&
            error("boundary weight 1 - $(bs.kb == 2.0 ? "2" : "")βv_h ≤ 0 at a face point ($(bs.kind) side)")
        1.0 - bs.kb * bf[i] * v_ex[i] <= 0.0 &&
            error("boundary weight 1 - $(bs.kb == 2.0 ? "2" : "")βg ≤ 0 for the boundary data g ($(bs.kind) side)")
    end
    return _abc_face(bs.gamma_abc, bs.eta_abc, bs.sigma, bs.kb, c2f, af, cf, bf, v_fp,
                     grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
end

"""
    Workspace(pre, N)

Preallocated arrays for [`edg_rhs!`](@ref) and [`rk4_step!`](@ref) on an `N×N`
mesh with the precomputed data `pre`.
"""
function Workspace(pre, N)
    nu = pre.qu1;  nv = pre.qv1;  nq = pre.qu1;  qh = pre.q_high
    z3(n) = zeros(n, N, N)
    (U = zeros(nu, nu, N, N), Ut = zeros(nu, nu, N, N),
     V = zeros(nv, nv, N, N), Vt = zeros(nv, nv, N, N),
     tu = zeros(1, nu * N * N), tv = zeros(1, nv * N * N),
     # element traces at the face Gauss points: u-normal derivatives (reference
     # scaling), v values and v-normal derivatives, on the sides B, T, L, R
     duB = z3(nq), duT = z3(nq), duL = z3(nq), duR = z3(nq),
     vB  = z3(nq), vT  = z3(nq), vL  = z3(nq), vR  = z3(nq),
     dvB = z3(nq), dvT = z3(nq), dvL = z3(nq), dvR = z3(nq),
     # per element and side: A (lifted with the trace of φ_v, scale 2/h),
     # D (lifted with the normal derivative of φ_v, scale 4/h²) and
     # Fu (lifted with the normal derivative of φ_u, u-equation)
     AB = z3(nq), AT = z3(nq), AL = z3(nq), AR = z3(nq),
     DB = z3(nq), DT = z3(nq), DL = z3(nq), DR = z3(nq),
     FB = z3(nq), FT = z3(nq), FL = z3(nq), FR = z3(nq),
     Xw = zeros(nq, N * N), Yv = zeros(nv, N * N), Yu = zeros(nu, N * N),
     Rv = zeros(nv, nv, N, N), Fu = zeros(nu, nu, N, N),
     UT = zeros(nu, nu, N, N), VT = zeros(nv, nv, N, N),
     bu = zeros(nu * nu), bv = zeros(nv * nv),
     # conjugate-gradient work arrays (one element at a time)
     cg_T1 = zeros(qh, nv), cg_T2 = zeros(nv, qh), cg_Y = zeros(qh, qh), cg_G = zeros(qh, qh),
     cg_r = zeros(nv, nv), cg_p = zeros(nv, nv), cg_Ap = zeros(nv, nv),
     # outputs in the state layout and RK4 stages
     ut = zeros(N, N, nu, nu), vt = zeros(N, N, nv, nv),
     k1u = zeros(N, N, nu, nu), k2u = zeros(N, N, nu, nu), k3u = zeros(N, N, nu, nu), k4u = zeros(N, N, nu, nu),
     k1v = zeros(N, N, nv, nv), k2v = zeros(N, N, nv, nv), k3v = zeros(N, N, nv, nv), k4v = zeros(N, N, nv, nv),
     su = zeros(N, N, nu, nu), sv = zeros(N, N, nv, nv))
end

# out[i, j, k] = Σ_b P[i, b] Σ_a E[a] A[a, b, j, k]: trace on the side selected by
# the row vector E, evaluated at the face Gauss points.
function _trace!(out, P, E, A, tmp)
    n = size(A, 1)
    mul!(tmp, E, reshape(A, n, :))
    mul!(reshape(out, size(out, 1), :), P, reshape(tmp, n, :))
    return out
end

# R[a, b, j, k] += c E[a] Y[b, j, k]   (horizontal sides)  or
# R[a, b, j, k] += c Y[a, j, k] E[b]   (vertical sides),   with Y = P' (w ∘ Z).
function _lift!(R, P, w, Z, E, c, horizontal::Bool, Xw, Y)
    nq = size(Z, 1);  n = size(P, 2);  N = size(R, 3)
    Xw .= w .* reshape(Z, nq, :)
    mul!(Y, transpose(P), Xw)
    Y4 = reshape(Y, n, N, N)
    Ev = vec(E)
    if horizontal
        R .+= c .* reshape(Ev, n, 1, 1, 1) .* reshape(Y4, 1, n, N, N)
    else
        R .+= c .* reshape(Y4, n, 1, N, N) .* reshape(Ev, 1, n, 1, 1)
    end
    return R
end

"""
    edg_rhs!(ws, u, v, t, N, h, pre) -> (ws.ut, ws.vt)

Evaluate the semi-discrete EDG right-hand side at time `t` into the workspace
`ws` (see [`Workspace`](@ref)). The returned arrays are overwritten by the next
call.
"""
function edg_rhs!(ws, u::Array{Float64,4}, v::Array{Float64,4},
                  t::Float64, N::Int, h::Float64, pre::NamedTuple)
    nu = pre.qu1;  nv = pre.qv1;  omega = pre.omega;  w = pre.w
    P_u = pre.P_u;  P_v = pre.P_v
    theta = pre.theta;  tau = pre.tau;  lu = pre.lambda_u;  lv = pre.lambda_v
    two_h = 2.0 / h;  four_hh = 4.0 / (h * h)
    # time factors of the manufactured solution u = cos(ωt) Φ(x); both are
    # switched off outside the manufactured-solution mode, which leaves only
    # the u₀ terms of the ũ = u - u₀ change of variables.
    ct = pre.mms_on ? cos(omega * t) : 0.0
    st = pre.mms_on ? sin(omega * t) : 0.0

    # ---- 1. element-last copies (and per-element transposes) ----
    U = ws.U;  V = ws.V;  Ut = ws.Ut;  Vt = ws.Vt
    permutedims!(U, u, (3, 4, 1, 2));  permutedims!(Ut, u, (4, 3, 1, 2))
    permutedims!(V, v, (3, 4, 1, 2));  permutedims!(Vt, v, (4, 3, 1, 2))

    # ---- 2. traces: bottom/top contract the y-mode, left/right the x-mode ----
    _trace!(ws.duB, P_u, pre.E1bl_u, U, ws.tu);   _trace!(ws.duT, P_u, pre.E1br_u, U, ws.tu)
    _trace!(ws.duL, P_u, pre.E1bl_u, Ut, ws.tu);  _trace!(ws.duR, P_u, pre.E1br_u, Ut, ws.tu)
    _trace!(ws.vB,  P_v, pre.Ebl_v,  V, ws.tv);   _trace!(ws.vT,  P_v, pre.Ebr_v,  V, ws.tv)
    _trace!(ws.vL,  P_v, pre.Ebl_v,  Vt, ws.tv);  _trace!(ws.vR,  P_v, pre.Ebr_v,  Vt, ws.tv)
    _trace!(ws.dvB, P_v, pre.E1bl_v, V, ws.tv);   _trace!(ws.dvT, P_v, pre.E1br_v, V, ws.tv)
    _trace!(ws.dvL, P_v, pre.E1bl_v, Vt, ws.tv);  _trace!(ws.dvR, P_v, pre.E1br_v, Vt, ws.tv)

    # ---- 3. face fluxes ----
    # interior faces: s = v^- - v^+ is the jump seen from the element on the
    # "+" side (above, resp. to the right); g* = θ ∂_n w^+ + (1-θ) ∂_n w^-
    # (reference scaling), jg = (2/h)(∂ w^- - ∂ w^+).
    _interior_faces!(ws.AB, ws.DB, ws.FB, ws.AT, ws.DT, ws.FT,
                     ws.vT, ws.vB, ws.duT, ws.duB, ws.dvT, ws.dvB,
                     pre.hc2, pre.haf, theta, tau, lu, lv, two_h, N, Val(:horizontal))
    _interior_faces!(ws.AL, ws.DL, ws.FL, ws.AR, ws.DR, ws.FR,
                     ws.vR, ws.vL, ws.duR, ws.duL, ws.dvR, ws.dvL,
                     pre.vc2, pre.vaf, theta, tau, lu, lv, two_h, N, Val(:vertical))
    _boundary_faces!(ws, pre, ct, st, omega, two_h, N)

    # ---- 4./5. element residuals: volume, source, lifted face terms ----
    Rv = ws.Rv;  Fu = ws.Fu;  bv = ws.bv
    @inbounds for k in 1:N, j in 1:N
        Ue = vec(view(U, :, :, j, k));  Ve = vec(view(V, :, :, j, k))
        mul!(bv, pre.Mc2_uv[j, k], Ue)
        mul!(bv, pre.Ma_vv[j, k], Ve, 1.0, 1.0)
        view(Rv, :, :, j, k) .= reshape(bv, nv, nv) .* (-four_hh)
    end
    # manufactured source: the projection of
    #   (1 - 2βv_ex) ∂_t v_ex - ∇·(c²∇u_ex) - ∇·(α∇v_ex) + ∇·(c²∇u₀),
    # u_ex = cos(ωt)Φ, as a combination of time-independent projections
    Rv .+= (-omega^2 * ct) .* pre.PiSP .+ (-2.0 * omega^3 * st * ct) .* pre.PiBSP2 .+
           (1.0 - ct) .* pre.PiCdiv .+ (omega * st) .* pre.PiDdiv
    if pre.src_on
        Rv .+= pre.src_time(t) .* pre.src_projE
    end
    fill!(Fu, 0.0)
    for (A, D, F, Ev, E1v, E1u, hz) in ((ws.AB, ws.DB, ws.FB, pre.Ebl_v, pre.E1bl_v, pre.E1bl_u, true),
                                        (ws.AT, ws.DT, ws.FT, pre.Ebr_v, pre.E1br_v, pre.E1br_u, true),
                                        (ws.AL, ws.DL, ws.FL, pre.Ebl_v, pre.E1bl_v, pre.E1bl_u, false),
                                        (ws.AR, ws.DR, ws.FR, pre.Ebr_v, pre.E1br_v, pre.E1br_u, false))
        _lift!(Rv, P_v, w, A, Ev,  two_h,   hz, ws.Xw, ws.Yv)
        _lift!(Rv, P_v, w, D, E1v, four_hh, hz, ws.Xw, ws.Yv)
        _lift!(Fu, P_u, w, F, E1u, 1.0,     hz, ws.Xw, ws.Yu)
    end

    # u-equation: (c²∇φ_u, ∇u̇_h) + mean condition, pinned c²-stiffness (Cholesky)
    bu = ws.bu;  UT = ws.UT
    @inbounds for k in 1:N, j in 1:N
        Ve = vec(view(V, :, :, j, k))
        mul!(bu, transpose(pre.Mc2_uv[j, k]), Ve)
        bu .-= vec(view(Fu, :, :, j, k))
        bu[1] += V[1, 1, j, k]
        R = pre.R_ut[j, k]
        ldiv!(transpose(R), bu);  ldiv!(R, bu)
        vec(view(UT, :, :, j, k)) .= bu
    end

    # ---- 6. mass system ----
    _mass_solve_cg!(ws.VT, V, Rv, pre, ws, N)

    permutedims!(ws.ut, UT, (3, 4, 1, 2))
    permutedims!(ws.vt, ws.VT, (3, 4, 1, 2))
    return ws.ut, ws.vt
end

"""
    edg_rhs(u, v, t, N, h, pre) -> (u_t, v_t)

Evaluate the semi-discrete EDG right-hand side at time `t` (allocating
version of [`edg_rhs!`](@ref)).
"""
function edg_rhs(u::Array{Float64,4}, v::Array{Float64,4},
                 t::Float64, N::Int, h::Float64, pre::NamedTuple)
    ws = Workspace(pre, N)
    edg_rhs!(ws, u, v, t, N, h, pre)
    return copy(ws.ut), copy(ws.vt)
end

# Interior faces between the elements on the "-" side (below / to the left)
# and the "+" side (above / to the right). For horizontal faces the face
# index runs over (point, row of the "+" element, column); for vertical faces
# over (point, row, column of the "+" element). Writes the A, D, F arrays of
# the "+" element's B/L side and of the "-" element's T/R side.
function _interior_faces!(Ap, Dp, Fp, Am, Dm, Fm, vM, vP, duM, duP, dvM, dvP,
                          c2f, aff, theta, tau, lu, lv, two_h, N, ::Val{orient}) where {orient}
    N < 2 && return nothing
    if orient === :horizontal
        P = (:, 2:N, :);  M = (:, 1:N-1, :);  Fc = (:, 2:N, :)
    else
        P = (:, :, 2:N);  M = (:, :, 1:N-1);  Fc = (:, :, 2:N)
    end
    @views begin
        vm = vM[M...];   vp = vP[P...]
        dum = duM[M...]; dup = duP[P...]
        dvm = dvM[M...]; dvp = dvP[P...]
        c2 = c2f[Fc...]; af = aff[Fc...]
        # "+" element (bottom / left side of its own)
        @. Ap[P...] = (c2 * lu + af * lv) * (vm - vp) -
                      two_h * (c2 * (theta * dup + (1 - theta) * dum) + af * (theta * dvp + (1 - theta) * dvm))
        @. Dp[P...] = -af * (theta * (vm - vp) - tau * (c2 * (two_h * (dum - dup)) + af * (two_h * (dvm - dvp))))
        @. Fp[P...] =  c2 * (theta * (vm - vp) - tau * (c2 * (two_h * (dum - dup)) + af * (two_h * (dvm - dvp))))
        # "-" element (top / right side of its own); its jump is -(vm - vp)
        @. Am[M...] = two_h * (c2 * (theta * dup + (1 - theta) * dum) + af * (theta * dvp + (1 - theta) * dvm)) -
                      (c2 * lu + af * lv) * (vm - vp)
        @. Dm[M...] =  af * ((1 - theta) * (vp - vm) - tau * (c2 * (two_h * (dum - dup)) + af * (two_h * (dvm - dvp))))
        @. Fm[M...] = -c2 * ((1 - theta) * (vp - vm) - tau * (c2 * (two_h * (dum - dup)) + af * (two_h * (dvm - dvp))))
    end
    return nothing
end

# Physical boundary faces; side s uses the kind pre.bc[s] with exact data in the
# manufactured-solution mode, -(c²∇u₀)·n in :initial_value mode and zero data in
# :source mode.
function _boundary_faces!(ws, pre, ct, st, omega, two_h, N)
    @views begin
        # bottom (y = 0, n = (0,-1)) and top (y = 1, n = (0,+1)): rows 1 and N, along x
        for (bs, row, frow, sgn, sp, spn, A, D, F, vtr, dutr, dvtr) in
                ((pre.bc.bot, 1, 1,     -1.0, pre.bcB_sp, pre.bcB_spy, ws.AB, ws.DB, ws.FB, ws.vB, ws.duB, ws.dvB),
                 (pre.bc.top, N, N + 1,  1.0, pre.bcT_sp, pre.bcT_spy, ws.AT, ws.DT, ws.FT, ws.vT, ws.duT, ws.dvT))
            c2f = pre.hc2[:, frow, :];  af = pre.haf[:, frow, :]
            cf  = pre.hcf[:, frow, :];  bf = pre.hbf[:, frow, :]
            grad_u_n = sgn .* (two_h .* dutr[:, row, :])
            grad_v_n = sgn .* (two_h .* dvtr[:, row, :])
            gu_ex = sgn .* (ct .* spn .- spn)              # ∇(u - u₀)·n of the exact solution
            gv_ex = sgn .* (-omega * st .* spn)
            v_ex  = -omega * st .* sp
            F1, dv = _side_face(bs, c2f, af, cf, bf, vtr[:, row, :], grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
            A[:, row, :] .= F1
            D[:, row, :] .= sgn .* (af .* dv)
            F[:, row, :] .= -sgn .* (c2f .* dv)
        end
        # left (x = 0, n = (-1,0)) and right (x = 1, n = (+1,0)): columns 1 and N, along y
        for (bs, col, fcol, sgn, sp, spn, A, D, F, vtr, dutr, dvtr) in
                ((pre.bc.lft, 1, 1,     -1.0, pre.bcL_sp, pre.bcL_spx, ws.AL, ws.DL, ws.FL, ws.vL, ws.duL, ws.dvL),
                 (pre.bc.rgt, N, N + 1,  1.0, pre.bcR_sp, pre.bcR_spx, ws.AR, ws.DR, ws.FR, ws.vR, ws.duR, ws.dvR))
            c2f = pre.vc2[:, :, fcol];  af = pre.vaf[:, :, fcol]
            cf  = pre.vcf[:, :, fcol];  bf = pre.vbf[:, :, fcol]
            grad_u_n = sgn .* (two_h .* dutr[:, :, col])
            grad_v_n = sgn .* (two_h .* dvtr[:, :, col])
            gu_ex = sgn .* (ct .* spn .- spn)
            gv_ex = sgn .* (-omega * st .* spn)
            v_ex  = -omega * st .* sp
            F1, dv = _side_face(bs, c2f, af, cf, bf, vtr[:, :, col], grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
            A[:, :, col] .= F1
            D[:, :, col] .= sgn .* (af .* dv)
            F[:, :, col] .= -sgn .* (c2f .* dv)
        end
    end
    return nothing
end

# Mass system of the v-equation on every element,
#     M(v_h) X = R,   M(v_h) = Φ' diag((1 - 2β v_q) W) Φ,   Φ = P ⊗ P,
# with P the 1D Legendre basis at the q_high Gauss points, W the tensor
# weights and v_q = v_h at those points. The basis is orthonormal and the rule
# integrates the products exactly, so Φ' diag(W) Φ = I and
#     M(v_h) = I - K(v_h),   K X = P' (G ∘ (P X P')) P,   G = 2β v_q W.
# Since W^{1/2}Φ has orthonormal columns, the eigenvalues of M lie in
# [min_q (1 - 2β v_q), max_q (1 - 2β v_q)]: M is symmetric positive definite
# whenever 1 - 2β v_q > 0 at all points, with condition number at most
# max(1 - 2βv_q)/min(1 - 2βv_q). The system is solved by
# conjugate gradients, matrix-free and started from X = R (the exact solution
# for K = 0); relative residual tolerance pre.mass_tol.
function _mass_solve_cg!(X, V, Rv, pre, ws, N)
    P = pre.P_h_v;  Pt = pre.P_h_vT;  W = pre.W2d_h;  tol = pre.mass_tol
    nv = pre.qv1;  maxit = nv * nv + 10
    T1 = ws.cg_T1;  T2 = ws.cg_T2;  Y = ws.cg_Y;  G = ws.cg_G
    r = ws.cg_r;  p = ws.cg_p;  Ap = ws.cg_Ap
    applyK!(out, Z) = (mul!(T1, P, Z); mul!(Y, T1, Pt); Y .*= G; mul!(T2, Pt, Y); mul!(out, T2, P); out)
    nfail = 0
    @inbounds for k in 1:N, j in 1:N
        Ve = view(V, :, :, j, k);  Re = view(Rv, :, :, j, k);  Xe = view(X, :, :, j, k)
        mul!(T1, P, Ve);  mul!(Y, T1, Pt)                       # v_q
        G .= 2.0 .* view(pre.beta_hE, :, :, j, k) .* Y .* W
        Xe .= Re
        applyK!(r, Xe)                                          # r = R - M X = K R
        p .= r
        rr = dot(r, r);  r0 = sqrt(dot(Re, Re));  it = 0
        while sqrt(rr) > tol * r0 && it < maxit
            it += 1
            applyK!(Ap, p);  Ap .= p .- Ap                      # Ap = M p
            pAp = dot(p, Ap)
            pAp > 0 || error("mass matrix not positive definite on element ($j, $k): 1 - 2βv_h ≤ 0")
            a = rr / pAp
            Xe .+= a .* p;  r .-= a .* Ap
            rrn = dot(r, r);  p .= r .+ (rrn / rr) .* p;  rr = rrn
        end
        sqrt(rr) > tol * r0 && (nfail += 1)
    end
    nfail > 0 && @warn "mass solve: $nfail element systems did not reach the tolerance $tol"
    return X
end
