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
# (see `solve`), which only adds the known terms ∇·(c²∇u₀) to the source,
# -(c²∇u₀)·n to the boundary data of the flux c²∇u·n (absorbing and wall
# sides) and, where c²∇u₀·n jumps across an interior face, the jump
# [[c²∇u₀]] to the face terms (`_interior_faces_os!`).
#
# The coefficients may jump across mesh faces: on a face every element uses
# its own values of c², α (and β on the boundary), precomputed in solver.jl.
#
# Structure of one evaluation (all arrays are preallocated in a `Workspace`):
#   1. copy u, v to element-last arrays U[:, :, j, k] (and their transposes);
#   2. traces on the four sides of every element (values and normal
#      derivatives at the face Gauss points);
#   3. numerical fluxes on all faces; every interior face is visited once and
#      writes the contributions to both neighbouring elements into the slots
#      of their sides (A, D, F of the side B, T, L or R), one writer per slot;
#   4. lift of the face contributions back to the elements;
#   5. volume terms matrix-free (a few large matrix products per block), the
#      manufactured source, and the u-equation element by element;
#   6. the mass system ∫(1 - 2βv_h) φ v̇_h = R on every element, matrix-free,
#      by Chebyshev iteration (`_cheb_setup!`, `_cheb_iterate!`) or
#      element-by-element conjugate gradients (`_mass_cg_block!`), selected by
#      pre.mass_solver.
# The result is that of the element-by-element version of v0.1.0 (kept in
# test/reference_rhs.jl, dense LU mass solve; the boundary kinds added later
# are compared with it through equivalent parameters) up to the order of
# floating-point operations and the tolerance pre.mass_tol = 1e-14 of the
# mass solve.
#
# Multithreading. The elements (linear index e = j + (k-1)N) are split into
# ws.nb contiguous blocks, each with its own work arrays, and one evaluation
# runs in three parallel regions, one task per block, separated by barriers:
#   region 1 (element blocks): steps 1 and 2, and the spectral interval and
#            step count of the Chebyshev iteration on every element;
#   region 2 (blocks of mesh columns): step 3; every slot is written by
#            exactly one face and every face by one block (the vertical faces
#            at the left edge of a column block write the R slots of the
#            column before it, which no other block writes), so the blocks
#            write disjoint data;
#   region 3 (element blocks): steps 4 to 6 and the copy to the state layout.
# All elements take the same number of Chebyshev steps, the maximum over all
# elements (taken between regions 1 and 3), and every other operation on an
# element is the same in every block. The result therefore does not depend on
# the number of blocks, bit for bit, given a BLAS that computes each entry of
# a product independently of the size of the block (OpenBLAS does; blocks of
# at least 4 elements avoid Julia's own 2×2 and 3×3 products). Work that would
# take many small BLAS calls per block (traces, lifts, conjugate gradients) is
# written as plain loops that sum in the same order as BLAS, because small
# BLAS calls from parallel tasks compete for the library's internal buffers.
# A non-positive boundary weight or mass matrix found in a region is reported
# after it, as the first one in the order of the serial loops.

# Pointwise fluxes on a physical boundary face. Inputs are values at the face
# quadrature points (arrays of any common shape, or scalars); `*_ex` are the
# boundary data. Every kind returns (F1, dv): the total numerical normal flux
# ((c²∇u)* + (α∇v)*)·n entering the v-equation and the v-trace correction
# v* - v_h. The lift of (F1, dv) is the same for all kinds.

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

# Flux on one side with the parameters `bs` of its boundary kind, without the
# check of the weight.
@inline _side_flux(bs, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex) =
    bs.kind === :wall ? _wall_face(bs.kappa, c2f, af, cf, grad_u_n, grad_v_n, gu_ex, gv_ex) :
                        _abc_face(bs.gamma_abc, bs.eta_abc, bs.sigma, bs.kb, c2f, af, cf, bf, v_fp,
                                  grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)

# The weight needs 1 - kβv > 0 at the face points for the trace v_h and the
# data g (the mass solve only sees the volume points): 0 if both hold, else 1
# (trace) or 2 (data). A NaN passes these comparisons and propagates (in
# :source mode the blow-up check stops it).
@inline _weight_check(kb, bf, v_fp, v_ex) =
    1.0 - kb * bf * v_fp <= 0.0 ? 1 : 1.0 - kb * bf * v_ex <= 0.0 ? 2 : 0

function _weight_error(bs, check)
    k = bs.kb == 2.0 ? "2" : ""
    check == 1 && error("boundary weight 1 - $(k)βv_h ≤ 0 at a face point ($(bs.kind) side)")
    error("boundary weight 1 - $(k)βg ≤ 0 for the boundary data g ($(bs.kind) side)")
end

# Flux on one side for arrays of face values, after checking the weight at all
# points (the RHS checks and evaluates point by point in `_boundary_side!`).
function _side_face(bs, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
    if bs.kind !== :wall
        @inbounds for i in eachindex(bf, v_fp, v_ex)
            check = _weight_check(bs.kb, bf[i], v_fp[i], v_ex[i])
            check == 0 || _weight_error(bs, check)
        end
    end
    return _side_flux(bs, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
end

# 1:n split into k contiguous ranges whose lengths differ by at most one
_chunks(n, k) = [_chunk(n, k, c) for c in 1:k]
_chunk(n, k, c) = ((c - 1) * n ÷ k + 1):(c * n ÷ k)

# Default number of blocks for the work w = N²(q_u+1)² of a mesh: a parallel
# region costs a fixed few tens of microseconds of synchronization, so small
# meshes run serially and medium ones on at most 4 blocks. The thresholds are
# roughly where, per RK4 step, 4 blocks became faster than 1 and 8 blocks
# (8 threads) faster than 4, on an Apple M4 Pro for q_u = 1, 2, 3, 4, 6.
function _default_nblocks(pre, N)
    w = N * N * pre.qu1^2;  nt = Threads.nthreads()
    return w < 5000 ? 1 : w < 20000 ? min(4, nt) : nt
end

# Work arrays of a block of Eb elements, in the layout (mode, element, mode):
# volume terms (standard points nq), Chebyshev iteration (the interval data
# θ, δ, σ1, ρ per element); and of one element: conjugate gradients, lifts,
# the right-hand side of the u-equation.
function _block_workspace(pre, Eb)
    nu = pre.qu1;  nv = pre.qv1;  nq = pre.qu1;  qh = pre.q_high
    (vl_U = zeros(nu, Eb, nu), vl_V = zeros(nv, Eb, nv),
     vl_G1 = zeros(nq, Eb, nq), vl_G2 = zeros(nq, Eb, nq),
     vl_H1 = zeros(nq, Eb, nq), vl_H2 = zeros(nq, Eb, nq),
     vl_Tu = zeros(nq, Eb, nu), vl_Tv = zeros(nq, Eb, nv),
     vl_Su = zeros(nu, Eb, nq), vl_Sv = zeros(nv, Eb, nq),
     vl_R = zeros(nv, Eb, nv), vl_B = zeros(nu, Eb, nu),
     ch_Z = zeros(nv, Eb, nv), ch_r = zeros(nv, Eb, nv), ch_d = zeros(nv, Eb, nv),
     ch_Kd = zeros(nv, Eb, nv), ch_X = zeros(nv, Eb, nv), ch_S = zeros(nv, Eb, qh),
     ch_T = zeros(qh, Eb, nv), ch_Y = zeros(qh, Eb, qh), ch_G = zeros(qh, Eb, qh),
     ch_th = zeros(1, Eb, 1), ch_de = zeros(1, Eb, 1), ch_s1 = zeros(1, Eb, 1),
     ch_rho = zeros(1, Eb, 1), ch_rhon = zeros(1, Eb, 1),
     cg_T1 = zeros(qh, nv), cg_T2 = zeros(nv, qh), cg_Y = zeros(qh, qh), cg_G = zeros(qh, qh),
     cg_r = zeros(nv, nv), cg_p = zeros(nv, nv), cg_Ap = zeros(nv, nv),
     lf_Y = zeros(nu, 3), ue_b = zeros(nu * nu))
end

"""
    Workspace(pre, N; nblocks = nothing)

Preallocated arrays for [`edg_rhs!`](@ref) and [`rk4_step!`](@ref) on an `N×N`
mesh with the precomputed data `pre`.

The elements are split into `nblocks` contiguous blocks (at most `N²÷4`, so
that every block has at least 4 elements; the effective number is `ws.nb`),
each with its own work arrays. With more than one block, every evaluation
runs its blocks as parallel tasks (`Threads.@spawn`), which needs Julia
started with several threads (`julia -t 8`) to be faster; the results do not
depend on `nblocks`. `nblocks = 1` runs everything in the calling task. The
default `nothing` chooses from the size of the problem: one block per Julia
thread on large meshes, at most 4 blocks on medium ones and a single block on
small ones, where parallel tasks cost more than they save (see
*Multithreading* in the README). Keep BLAS at one thread while `ws.nb > 1`
(see [`edg_rhs!`](@ref)).
"""
function Workspace(pre, N; nblocks::Union{Nothing,Integer} = nothing)
    nblocks === nothing && (nblocks = _default_nblocks(pre, N))
    nblocks >= 1 || error("nblocks must be ≥ 1 (got $nblocks)")
    nu = pre.qu1;  nv = pre.qv1;  nq = pre.qu1;  E = N * N
    nb = min(Int(nblocks), max(1, E ÷ 4))
    eblk = _chunks(E, nb)
    cblk = _chunks(N, min(nb, N))
    z3(n) = zeros(n, N, N)
    (nb = nb, eblk = eblk, cblk = cblk,
     blk = [_block_workspace(pre, length(er)) for er in eblk],
     # per block: Chebyshev step count, first element with a non-positive mass
     # matrix, CG systems above the tolerance, finite entries after the last
     # RK4 step (`_rk4_final!`); per column block: first face point with a
     # non-positive boundary weight (`_boundary_key`), 0 if none
     nsteps = zeros(Int, nb), mfail = zeros(Int, nb), nfail = zeros(Int, nb), fin = ones(Bool, nb),
     bfail = zeros(Int, length(cblk)),
     U = zeros(nu, nu, N, N), Ut = zeros(nu, nu, N, N),
     V = zeros(nv, nv, N, N), Vt = zeros(nv, nv, N, N),
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
     Rv = zeros(nv, nv, N, N), Fu = zeros(nu, nu, N, N),
     UT = zeros(nu, nu, N, N), VT = zeros(nv, nv, N, N), VB = zeros(nu, nu, N, N),
     # outputs in the state layout and RK4 stages
     ut = zeros(N, N, nu, nu), vt = zeros(N, N, nv, nv),
     k1u = zeros(N, N, nu, nu), k2u = zeros(N, N, nu, nu), k3u = zeros(N, N, nu, nu), k4u = zeros(N, N, nu, nu),
     k1v = zeros(N, N, nv, nv), k2v = zeros(N, N, nv, nv), k3v = zeros(N, N, nv, nv), k4v = zeros(N, N, nv, nv),
     su = zeros(N, N, nu, nu), sv = zeros(N, N, nv, nv))
end

# Run f(b) for b = 1:n: in the calling task for n = 1, else as n parallel
# tasks. After all tasks have finished, the error of the first failed task in
# the order of b is rethrown: an ErrorException (the package's own messages)
# as it was raised, any other error as the TaskFailedException, which keeps
# the backtrace inside the task.
function _foreach_block(f::F, n::Int) where {F}
    if n == 1
        f(1)
        return nothing
    end
    # the tasks share one reference to f (a closure holding copies of the large
    # NamedTuples ws and pre) instead of one copy each
    fr = Ref(f)
    tasks = Vector{Task}(undef, n)
    for b in 1:n
        tasks[b] = Threads.@spawn fr[](b)
    end
    err = nothing
    for task in tasks
        try
            wait(task)
        catch e
            err === nothing && (err = _task_error(e))
        end
    end
    err === nothing || throw(err)
    return nothing
end
_task_error(e) = (inner = _task_exception(e); inner isa ErrorException ? inner : e)
_task_exception(e) = e isa TaskFailedException ? _task_exception(e.task.exception) :
                     e isa CompositeException ? _task_exception(first(e.exceptions)) : e

# Copies between the element-last arrays A[a, b, e] (A: (n, n, N, N)) and a
# block in the layout Z[a, e - e0, b] (Z: (n, Eb, n)), for e in er = e0+1:e0+Eb.
function _gather!(Z, A, er)
    n = size(Z, 1);  e0 = first(er) - 1
    @inbounds for b in 1:n, e in er, a in 1:n
        Z[a, e - e0, b] = A[a + n * (b - 1) + n * n * (e - 1)]
    end
    return Z
end
function _scatter!(A, Z, er)
    n = size(Z, 1);  e0 = first(er) - 1
    @inbounds for e in er, b in 1:n, a in 1:n
        A[a + n * (b - 1) + n * n * (e - 1)] = Z[a, e - e0, b]
    end
    return A
end

# Elements er of the state x (layout (N, N, n, n), x[e, a, b]) to the
# element-last arrays X[a, b, e] and Xt[a, b, e] = x[e, b, a], and back.
function _from_state!(X, Xt, x, er)
    n = size(X, 1);  E = length(x) ÷ (n * n)
    @inbounds for e in er, b in 1:n, a in 1:n
        xe = x[e + E * (a - 1) + E * n * (b - 1)]
        X[a + n * (b - 1) + n * n * (e - 1)] = xe
        Xt[b + n * (a - 1) + n * n * (e - 1)] = xe
    end
    return X
end
function _to_state!(x, X, er)
    n = size(X, 1);  E = length(x) ÷ (n * n)
    @inbounds for e in er, b in 1:n, a in 1:n
        x[e + E * (a - 1) + E * n * (b - 1)] = X[a + n * (b - 1) + n * n * (e - 1)]
    end
    return x
end

# Contractions of all elements of a block at once for arrays in the layout
# (mode, element, mode), two matrix products each. Per element,
#     _contract_fwd!:  out = L Z R'          (Z: n×n, out: m×m, L, R: m×n),
#     _contract_bwd!:  out = L' H R + β out  (H: m×m, out: n×n, L, R: m×n);
# the transposed factor is passed explicitly (Rt = R', Lt = L'), and T, S are
# work arrays of size (m, E, n) and (n, E, m).
function _contract_fwd!(out, T, L, Rt, Z)
    n = size(Z, 1);  E = size(Z, 2);  m = size(L, 1)
    mul!(reshape(T, m, E * n), L, reshape(Z, n, E * n))
    mul!(reshape(out, m * E, m), reshape(T, m * E, n), Rt)
    return out
end
function _contract_bwd!(out, S, Lt, R, H, β = 0.0)
    m = size(H, 1);  E = size(H, 2);  n = size(Lt, 1)
    mul!(reshape(S, n, E * m), Lt, reshape(H, m, E * m))
    mul!(reshape(out, n * E, n), reshape(S, n * E, m), R, 1.0, β)
    return out
end

# Volume terms on the elements er, matrix-free with the (q_u+1)-point rule:
#     Rv = (c²∇φ_v, ∇u_h) + (α∇φ_v, ∇v_h),   VB = (c²∇φ_u, ∇v_h)
# in the reference scaling (Rv is multiplied by -4/h² by the caller). On an
# element, with P, S the 1D values and derivatives of the basis at the points
# (rows y, columns x) and C = c²W, A = αW at the points,
#     (c²∇φ_v, ∇u_h) = S_v' (C ∘ (S_u U P_u')) P_v + P_v' (C ∘ (P_u U S_u')) S_v,
# and likewise for the other two terms; the gradients of u_h and v_h at the
# points are computed once. `bk` holds the work arrays of the block.
function _volume_terms!(Rv, VB, U, V, pre, bk, er)
    Ua = bk.vl_U;  Va = bk.vl_V;  G1 = bk.vl_G1;  G2 = bk.vl_G2;  H1 = bk.vl_H1;  H2 = bk.vl_H2
    Cw = view(pre.Cw, :, er, :);  Aw = view(pre.Aw, :, er, :)
    _gather!(Ua, U, er);  _gather!(Va, V, er)
    # c²∇u_h at the points
    _contract_fwd!(G1, bk.vl_Tu, pre.S_u, pre.P_uT, Ua);  _contract_fwd!(G2, bk.vl_Tu, pre.P_u, pre.S_uT, Ua)
    H1 .= Cw .* G1;  H2 .= Cw .* G2
    # ∇v_h at the points; H = c²∇u_h + α∇v_h
    _contract_fwd!(G1, bk.vl_Tv, pre.S_v, pre.P_vT, Va);  _contract_fwd!(G2, bk.vl_Tv, pre.P_v, pre.S_vT, Va)
    H1 .+= Aw .* G1;  H2 .+= Aw .* G2
    _contract_bwd!(bk.vl_R, bk.vl_Sv, pre.S_vT, pre.P_v, H1);  _contract_bwd!(bk.vl_R, bk.vl_Sv, pre.P_vT, pre.S_v, H2, 1.0)
    # c²∇v_h, tested with ∇φ_u
    G1 .*= Cw;  G2 .*= Cw
    _contract_bwd!(bk.vl_B, bk.vl_Su, pre.S_uT, pre.P_u, G1);  _contract_bwd!(bk.vl_B, bk.vl_Su, pre.P_uT, pre.S_u, G2, 1.0)
    _scatter!(Rv, bk.vl_R, er);  _scatter!(VB, bk.vl_B, er)
    return nothing
end

# Volume terms scaled by -4/h², plus the manufactured source on the elements
# er: the projection of
#   (1 - 2βv_ex) ∂_t v_ex - ∇·(c²∇u_ex) - ∇·(α∇v_ex) + ∇·(c²∇u₀),
# u_ex = cos(ωt)Φ, as a combination of time-independent projections, and the
# source term of :source mode with the time factor srct.
function _scale_source!(Rv, pre, er, ct, st, four_hh, srct)
    omega = pre.omega;  n2 = pre.qv1^2
    lr = (n2 * (first(er) - 1) + 1):(n2 * last(er))
    Rb = view(Rv, lr)
    Rb .= (-four_hh) .* Rb .+ (-omega^2 * ct) .* view(pre.PiSP, lr) .+ (-2.0 * omega^3 * st * ct) .* view(pre.PiBSP2, lr) .+
          (1.0 - ct) .* view(pre.PiCdiv, lr) .+ (omega * st) .* view(pre.PiDdiv, lr)
    if pre.src_on
        Rb .+= srct .* view(pre.src_projE, lr)
    end
    return nothing
end

# u-equation on the elements er: A_e u̇_e = b_e with A_e the c²-stiffness whose
# constant mode is pinned to the mean condition and
#     b_e = (c²∇φ_u, ∇v_h) - (lifted face terms) + e₁ (mean of v_h),
# applied with the stored inverse A_e⁻¹ in a plain loop over the elements, with
# no library call per element. A_e⁻¹ is symmetric and only its upper triangle
# is stored, packed by columns (entry (i, j), i ≤ j, in row j(j-1)/2 + i,
# element e in column e), so the loop reads half the data of the full inverse:
# column j adds A_e⁻¹[i, j] b_j to the rows i < j and the dot product of the
# same entries with b to row j.
function _u_equation!(UT, VB, Fu, V, Ainv, bk, er)
    E = size(Ainv, 2);  b = bk.ue_b;  n = length(b)
    X = reshape(UT, n, E);  B = reshape(VB, n, E);  F = reshape(Fu, n, E)
    V1 = reshape(V, :, E)
    @inbounds for e in er
        for i in 1:n
            b[i] = B[i, e] - F[i, e]
            X[i, e] = 0.0
        end
        b[1] += V1[1, e]
        for j in 1:n
            bj = b[j];  off = (j * (j - 1)) >> 1;  s = 0.0
            @simd for i in 1:j-1
                a = Ainv[off + i, e]
                X[i, e] = muladd(a, bj, X[i, e])
                s = muladd(a, b[i], s)
            end
            X[j, e] = muladd(Ainv[off + j, e], bj, X[j, e] + s)
        end
    end
    return UT
end

# Traces of the elements er on the four sides at the face Gauss points:
#     oB[i, e] = Σ_b P[i, b] Σ_a El[a] X[a, b, e],   oT[i, e] the same with Er,
# and oL, oR the same with Xt (bottom/top contract the y-mode of X, left/right
# the x-mode), every sum taken in order.
function _trace4!(oB, oT, oL, oR, P, El, Er, X, Xt, er)
    nq, n = size(P)
    @inbounds for e in er
        o = nq * (e - 1);  x0 = n * n * (e - 1)
        for i in 1:nq
            oB[o + i] = 0.0;  oT[o + i] = 0.0;  oL[o + i] = 0.0;  oR[o + i] = 0.0
        end
        for b in 1:n
            sB = 0.0;  sT = 0.0;  sL = 0.0;  sR = 0.0
            for a in 1:n
                x = X[x0 + a + n * (b - 1)];  xt = Xt[x0 + a + n * (b - 1)]
                sB = muladd(El[a], x, sB);   sT = muladd(Er[a], x, sT)
                sL = muladd(El[a], xt, sL);  sR = muladd(Er[a], xt, sR)
            end
            for i in 1:nq
                p = P[i, b]
                oB[o + i] = muladd(p, sB, oB[o + i]);  oT[o + i] = muladd(p, sT, oT[o + i])
                oL[o + i] = muladd(p, sL, oL[o + i]);  oR[o + i] = muladd(p, sR, oR[o + i])
            end
        end
    end
    return nothing
end

# Lifted face terms of the elements er, added to Rv (v-equation) and written
# to Fu (u-equation), side by side in the order B, T, L, R. On a side, with
# Y_Z[b] = Σ_i P[i, b] (w[i] Z[i]) (sum in order) for the face values
# Z ∈ {A, D, F} of the element and E the trace (A) or normal-derivative (D, F)
# row of the side,
#     Rv[a, b] += ((2/h) E_v[a]) Y_A[b],  then += ((4/h²) E1_v[a]) Y_D[b],
#     Fu[a, b] += (1 E1_u[a]) Y_F[b]
# on a horizontal side, and with (c Y[a]) E[b] in place of (c E[a]) Y[b] on a
# vertical side; bk.lf_Y holds Y_A, Y_D, Y_F.
function _lifts!(Rv, Fu, ws, pre, bk, er, two_h, four_hh)
    nv = pre.qv1;  nu = pre.qu1;  nq = pre.qu1
    P_v = pre.P_v;  P_u = pre.P_u;  w = pre.w;  Y = bk.lf_Y
    sides = ((ws.AB, ws.DB, ws.FB, pre.Ebl_v, pre.E1bl_v, pre.E1bl_u, true),
             (ws.AT, ws.DT, ws.FT, pre.Ebr_v, pre.E1br_v, pre.E1br_u, true),
             (ws.AL, ws.DL, ws.FL, pre.Ebl_v, pre.E1bl_v, pre.E1bl_u, false),
             (ws.AR, ws.DR, ws.FR, pre.Ebr_v, pre.E1br_v, pre.E1br_u, false))
    @inbounds for e in er
        r0 = nv * nv * (e - 1);  f0 = nu * nu * (e - 1);  z0 = nq * (e - 1)
        for k in 1:nu * nu
            Fu[f0 + k] = 0.0
        end
        for (A, D, F, Ev, E1v, E1u, hz) in sides
            for b in 1:nv
                sA = 0.0;  sD = 0.0
                for i in 1:nq
                    sA = muladd(P_v[i, b], w[i] * A[z0 + i], sA)
                    sD = muladd(P_v[i, b], w[i] * D[z0 + i], sD)
                end
                Y[b, 1] = sA;  Y[b, 2] = sD
            end
            for b in 1:nu
                sF = 0.0
                for i in 1:nq
                    sF = muladd(P_u[i, b], w[i] * F[z0 + i], sF)
                end
                Y[b, 3] = sF
            end
            if hz
                for b in 1:nv, a in 1:nv
                    k = r0 + a + nv * (b - 1)
                    Rv[k] = Rv[k] + (two_h * Ev[a]) * Y[b, 1]
                    Rv[k] = Rv[k] + (four_hh * E1v[a]) * Y[b, 2]
                end
                for b in 1:nu, a in 1:nu
                    k = f0 + a + nu * (b - 1)
                    Fu[k] = Fu[k] + (1.0 * E1u[a]) * Y[b, 3]
                end
            else
                for b in 1:nv, a in 1:nv
                    k = r0 + a + nv * (b - 1)
                    Rv[k] = Rv[k] + (two_h * Y[a, 1]) * Ev[b]
                    Rv[k] = Rv[k] + (four_hh * Y[a, 2]) * E1v[b]
                end
                for b in 1:nu, a in 1:nu
                    k = f0 + a + nu * (b - 1)
                    Fu[k] = Fu[k] + (1.0 * Y[a, 3]) * E1u[b]
                end
            end
        end
    end
    return nothing
end

"""
    edg_rhs!(ws, u, v, t, N, h, pre) -> (ws.ut, ws.vt)

Evaluate the semi-discrete EDG right-hand side at time `t` into the workspace
`ws` (see [`Workspace`](@ref)). The returned arrays are overwritten by the next
call.

With `ws.nb > 1` (the default `Workspace(pre, N)` on large enough meshes when
Julia has several threads) the evaluation runs as parallel tasks on the
element blocks of `ws`, with results identical to `ws.nb = 1`. The tasks make
their own BLAS calls, so set `BLAS.set_num_threads(1)` (`using LinearAlgebra`)
before calling `edg_rhs!` or [`rk4_step!`](@ref) with such a workspace
directly; [`solve`](@ref) and the allocating [`edg_rhs`](@ref) and
[`rk4_step`](@ref) do this themselves.
"""
function edg_rhs!(ws, u::Array{Float64,4}, v::Array{Float64,4},
                  t::Float64, N::Int, h::Float64, pre::NamedTuple)
    _edg_rhs!(ws.ut, ws.vt, ws, u, v, t, N, h, pre)
    return ws.ut, ws.vt
end

# The right-hand side into ut, vt (state layout). The kernels index with
# @inbounds, so the sizes of the state, the workspace and pre are checked first.
function _edg_rhs!(ut, vt, ws, u, v, t, N, h, pre)
    nu = pre.qu1;  nv = pre.qv1
    (size(u) == size(ut) == (N, N, nu, nu) && size(v) == size(vt) == (N, N, nv, nv) &&
     size(ws.U) == (nu, nu, N, N) && size(ws.V) == (nv, nv, N, N) &&
     size(ws.blk[1].ch_G, 1) == pre.q_high && size(pre.Ainv_u, 2) == N * N) ||
        throw(DimensionMismatch("state, workspace and pre do not match (N = $N, q_u + 1 = $nu, q_v + 1 = $nv)"))
    omega = pre.omega
    two_h = 2.0 / h;  four_hh = 4.0 / (h * h)
    # time factors of the manufactured solution u = cos(ωt) Φ(x); both are
    # switched off outside the manufactured-solution mode, which leaves only
    # the u₀ terms of the ũ = u - u₀ change of variables.
    ct = pre.mms_on ? cos(omega * t) : 0.0
    st = pre.mms_on ? sin(omega * t) : 0.0
    srct = pre.src_on ? pre.src_time(t) : 0.0
    cheb = pre.mass_solver === :chebyshev

    # ---- region 1: element-last copies, traces, Chebyshev intervals ----
    _foreach_block(ws.nb) do b
        er = ws.eblk[b]
        _from_state!(ws.U, ws.Ut, u, er);  _from_state!(ws.V, ws.Vt, v, er)
        _traces!(ws, pre, er)
        if cheb
            ws.nsteps[b], ws.mfail[b] = _cheb_setup!(ws.blk[b], ws.V, pre, er)
        end
    end

    # ---- region 2: face fluxes ----
    _foreach_block(length(ws.cblk)) do c
        kr = ws.cblk[c]
        _faces!(ws, pre, two_h, N, kr)
        ws.bfail[c] = _boundary_faces!(ws, pre, ct, st, two_h, N, kr)
    end
    _boundary_check(ws, pre, N)
    n = cheb ? _cheb_steps(ws) : 0

    # ---- region 3: volume, source, lifts, u-equation, mass system ----
    _foreach_block(ws.nb) do b
        er = ws.eblk[b];  bk = ws.blk[b]
        _volume_terms!(ws.Rv, ws.VB, ws.U, ws.V, pre, bk, er)
        _scale_source!(ws.Rv, pre, er, ct, st, four_hh, srct)
        _lifts!(ws.Rv, ws.Fu, ws, pre, bk, er, two_h, four_hh)
        # u-equation: (c²∇φ_u, ∇u̇_h) + mean condition, pinned c²-stiffness
        _u_equation!(ws.UT, ws.VB, ws.Fu, ws.V, pre.Ainv_u, bk, er)
        if cheb
            _cheb_iterate!(ws.VT, ws.Rv, pre, bk, er, n)
        else
            ws.nfail[b] = _mass_cg_block!(ws.VT, ws.V, ws.Rv, pre, bk, er, N)
        end
        _to_state!(ut, ws.UT, er);  _to_state!(vt, ws.VT, er)
    end
    cheb || _cg_report(ws, pre)
    return ut, vt
end

"""
    edg_rhs(u, v, t, N, h, pre; nblocks = nothing) -> (u_t, v_t)

Evaluate the semi-discrete EDG right-hand side at time `t` (allocating
version of [`edg_rhs!`](@ref) with `Workspace(pre, N; nblocks)`). With more
than one block, BLAS runs with one thread during the evaluation, as in
[`solve`](@ref).
"""
function edg_rhs(u::Array{Float64,4}, v::Array{Float64,4},
                 t::Float64, N::Int, h::Float64, pre::NamedTuple;
                 nblocks::Union{Nothing,Integer} = nothing)
    ws = Workspace(pre, N; nblocks = nblocks)
    counted = _blas_single!(ws)          # see solver.jl
    try
        edg_rhs!(ws, u, v, t, N, h, pre)
    finally
        _blas_restore(counted)
    end
    return copy(ws.ut), copy(ws.vt)
end

# Traces of the elements er: normal derivatives of u_h (reference scaling),
# values and normal derivatives of v_h.
function _traces!(ws, pre, er)
    _trace4!(ws.duB, ws.duT, ws.duL, ws.duR, pre.P_u, pre.E1bl_u, pre.E1br_u, ws.U, ws.Ut, er)
    _trace4!(ws.vB,  ws.vT,  ws.vL,  ws.vR,  pre.P_v, pre.Ebl_v,  pre.Ebr_v,  ws.V, ws.Vt, er)
    _trace4!(ws.dvB, ws.dvT, ws.dvL, ws.dvR, pre.P_v, pre.E1bl_v, pre.E1br_v, ws.V, ws.Vt, er)
    return nothing
end

# Interior faces in the mesh columns kr: s = v^- - v^+ is the jump seen from
# the element on the "+" side (above, resp. to the right);
# g* = θ ∂_n w^+ + (1-θ) ∂_n w^- (reference scaling), jg = (2/h)(∂ w^- - ∂ w^+).
# With one value of c² and α per face point (continuous coefficients) the
# faces are `_interior_faces!`; when some face has different coefficients on
# its two sides, or interface terms of u₀ (pre.one_sided), all faces are
# `_interior_faces_os!`.
function _faces!(ws, pre, two_h, N, kr)
    theta = pre.theta;  tau = pre.tau;  lu = pre.lambda_u;  lv = pre.lambda_v
    if pre.one_sided
        _interior_faces_os!(ws.AB, ws.DB, ws.FB, ws.AT, ws.DT, ws.FT,
                            ws.vT, ws.vB, ws.duT, ws.duB, ws.dvT, ws.dvB,
                            pre.hc2m, pre.hc2, pre.hafm, pre.haf, pre.hJC,
                            theta, tau, lu, lv, two_h, N, kr, Val(:horizontal))
        _interior_faces_os!(ws.AL, ws.DL, ws.FL, ws.AR, ws.DR, ws.FR,
                            ws.vR, ws.vL, ws.duR, ws.duL, ws.dvR, ws.dvL,
                            pre.vc2m, pre.vc2, pre.vafm, pre.vaf, pre.vJC,
                            theta, tau, lu, lv, two_h, N, kr, Val(:vertical))
        return nothing
    end
    _interior_faces!(ws.AB, ws.DB, ws.FB, ws.AT, ws.DT, ws.FT,
                     ws.vT, ws.vB, ws.duT, ws.duB, ws.dvT, ws.dvB,
                     pre.hc2, pre.haf, theta, tau, lu, lv, two_h, N, kr, Val(:horizontal))
    _interior_faces!(ws.AL, ws.DL, ws.FL, ws.AR, ws.DR, ws.FR,
                     ws.vR, ws.vL, ws.duR, ws.duL, ws.dvR, ws.dvL,
                     pre.vc2, pre.vaf, theta, tau, lu, lv, two_h, N, kr, Val(:vertical))
    return nothing
end

# Interior faces between the elements on the "-" side (below / to the left)
# and the "+" side (above / to the right), for the "+" elements in the mesh
# columns kr. For horizontal faces the face index runs over (point, row of the
# "+" element, column); for vertical faces over (point, row, column of the "+"
# element). Writes the A, D, F arrays of the "+" element's B/L side and of the
# "-" element's T/R side (for vertical faces the "-" element is in the column
# before; no other column range writes that slot). Both elements use the one
# value c2f, aff of c² and α at a face point, which is the paper's scheme for
# coefficients that are continuous across the face.
function _interior_faces!(Ap, Dp, Fp, Am, Dm, Fm, vM, vP, duM, duP, dvM, dvP,
                          c2f, aff, theta, tau, lu, lv, two_h, N, kr, ::Val{orient}) where {orient}
    N < 2 && return nothing
    if orient === :horizontal
        P = (:, 2:N, kr);  M = (:, 1:N-1, kr);  Fc = (:, 2:N, kr)
    else
        kp = max(first(kr), 2):last(kr)
        isempty(kp) && return nothing
        P = (:, :, kp);  M = (:, :, kp .- 1);  Fc = (:, :, kp)
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

# The same faces with one-sided coefficients: c²₋, α₋ of the "-" element
# (c2m, afm) and c²₊, α₊ of the "+" element (c2p, afp), the fluxes (2.9) of
# the paper with the paper's element 1 = "-" (n₁ = e) and element 2 = "+".
# With s = v⁻ - v⁺, the reference-scaled derivatives ∂ along e and
#     J = [[c²∇u]] + [[α∇v]] + K = (2/h)(c²₋ ∂u⁻ - c²₊ ∂u⁺ + α₋ ∂v⁻ - α₊ ∂v⁺) + K,
#     G = {{c²∇u + α∇v}}_{1-θ}·e = (2/h)((1-θ)(c²₋ ∂u⁻ + α₋ ∂v⁻) + θ(c²₊ ∂u⁺ + α₊ ∂v⁺)),
# the trace is v* = θ v⁻ + (1-θ) v⁺ - τ J and the penalty weight the mean
# ½(c²₋ + c²₊) λ_u + ½(α₋ + α₊) λ_v (the values of `_interior_faces!` for
# continuous coefficients). K = JC = [[c²∇u₀]] is the known interface term of
# the ũ = u - u₀ change of variables; it adds -θK to the flux of the "-"
# element and -(1-θ)K to that of the "+" element and vanishes where c²∇u₀ is
# continuous. The lifted terms c²∇φ_u·n (v* - v) and α∇φ_v·n (v* - v) of each
# element use its own coefficients, which the energy identity (2.10) pairs
# with its own fluxes.
function _interior_faces_os!(Ap, Dp, Fp, Am, Dm, Fm, vM, vP, duM, duP, dvM, dvP,
                             c2mf, c2pf, afmf, afpf, jcf,
                             theta, tau, lu, lv, two_h, N, kr, ::Val{orient}) where {orient}
    N < 2 && return nothing
    nq = size(Ap, 1)
    # "+" element (i, j, k) and face (i, j, k); "-" element (i, j - dj, k - dk)
    if orient === :horizontal
        rows = 2:N;  cols = kr;  dj = 1;  dk = 0
    else
        rows = 1:N;  cols = max(first(kr), 2):last(kr);  dj = 0;  dk = 1
    end
    @inbounds for k in cols, j in rows
        jm = j - dj;  km = k - dk
        for i in 1:nq
            vm = vM[i, jm, km];    vp = vP[i, j, k]
            dum = duM[i, jm, km];  dup = duP[i, j, k]
            dvm = dvM[i, jm, km];  dvp = dvP[i, j, k]
            c2m = c2mf[i, j, k];   c2p = c2pf[i, j, k]
            afm = afmf[i, j, k];   afp = afpf[i, j, k]
            K   = jcf[i, j, k]
            J   = two_h * (c2m * dum - c2p * dup + afm * dvm - afp * dvp) + K
            G   = two_h * ((1 - theta) * (c2m * dum + afm * dvm) + theta * (c2p * dup + afp * dvp))
            pen = (0.5 * (c2m + c2p)) * lu + (0.5 * (afm + afp)) * lv
            s   = vm - vp
            dvsp = theta * s - tau * J                  # v* - v⁺
            dvsm = (1 - theta) * (vp - vm) - tau * J    # v* - v⁻
            # "+" element (bottom / left side of its own)
            Ap[i, j, k] = pen * s - G - (1 - theta) * K
            Dp[i, j, k] = -afp * dvsp
            Fp[i, j, k] =  c2p * dvsp
            # "-" element (top / right side of its own)
            Am[i, jm, km] = G - pen * s - theta * K
            Dm[i, jm, km] =  afm * dvsm
            Fm[i, jm, km] = -c2m * dvsm
        end
    end
    return nothing
end

# Physical boundary faces of the elements in the mesh columns kr; side s uses
# the kind pre.bc[s] with exact data in the manufactured-solution mode,
# -(c²∇u₀)·n in :initial_value mode and zero data in :source mode. Returns 0,
# or the smallest `_boundary_key` of a point with a non-positive weight.
function _boundary_faces!(ws, pre, ct, st, two_h, N, kr)
    bc = pre.bc;  key = 0
    # bottom (y = 0, n = (0,-1)) and top (y = 1, n = (0,+1)): rows 1 and N, along x
    key = _boundary_side!(key, 1, bc.bot, ws.AB, ws.DB, ws.FB, ws.vB, ws.duB, ws.dvB,
                          pre.hc2, pre.haf, pre.hcf, pre.hbf, pre.bcB_sp, pre.bcB_spy,
                          1, 1, -1.0, kr, ct, st, pre.omega, two_h, N, Val(:horizontal))
    key = _boundary_side!(key, 2, bc.top, ws.AT, ws.DT, ws.FT, ws.vT, ws.duT, ws.dvT,
                          pre.hc2, pre.haf, pre.hcf, pre.hbf, pre.bcT_sp, pre.bcT_spy,
                          N, N + 1, 1.0, kr, ct, st, pre.omega, two_h, N, Val(:horizontal))
    # left (x = 0, n = (-1,0)) and right (x = 1, n = (+1,0)): columns 1 and N, along y
    if 1 in kr
        key = _boundary_side!(key, 3, bc.lft, ws.AL, ws.DL, ws.FL, ws.vL, ws.duL, ws.dvL,
                              pre.vc2, pre.vaf, pre.vcf, pre.vbf, pre.bcL_sp, pre.bcL_spx,
                              1, 1, -1.0, 1:N, ct, st, pre.omega, two_h, N, Val(:vertical))
    end
    if N in kr
        key = _boundary_side!(key, 4, bc.rgt, ws.AR, ws.DR, ws.FR, ws.vR, ws.duR, ws.dvR,
                              pre.vc2, pre.vaf, pre.vcf, pre.vbf, pre.bcR_sp, pre.bcR_spx,
                              N, N + 1, 1.0, 1:N, ct, st, pre.omega, two_h, N, Val(:vertical))
    end
    return key
end

# Point i of face m of side s (1 = bot, 2 = top, 3 = lft, 4 = rgt) with the
# failed check (1 = trace, 2 = data): ordered as the serial loops over the
# sides, their points and the two checks.
_boundary_key(s, i, m, check, nq, N) = 2 * ((s - 1) * nq * N + (i - 1) + (m - 1) * nq) + check

# One side, point by point: element slots (i, rc, m) on a horizontal side (row
# rc, columns m in ms), (i, m, rc) on a vertical side (column rc, rows m in
# ms), face coefficients at face row/column fc, data sp[i, m], spn[i, m]
# (normal derivative of Φ), outward normal component sgn.
function _boundary_side!(key, s, bs, A, D, F, vtr, dutr, dvtr, c2a, afa, cfa, bfa, sp, spn,
                         rc, fc, sgn, ms, ct, st, omega, two_h, N, ::Val{orient}) where {orient}
    nq = size(A, 1);  wall = bs.kind === :wall
    @inbounds for m in ms, i in 1:nq
        I  = orient === :horizontal ? CartesianIndex(i, rc, m) : CartesianIndex(i, m, rc)
        If = orient === :horizontal ? CartesianIndex(i, fc, m) : CartesianIndex(i, m, fc)
        c2f = c2a[If];  af = afa[If];  cf = cfa[If];  bf = bfa[If]
        v_fp = vtr[I]
        grad_u_n = sgn * (two_h * dutr[I])
        grad_v_n = sgn * (two_h * dvtr[I])
        gu_ex = sgn * (ct * spn[i, m] - spn[i, m])               # ∇(u - u₀)·n of the exact solution
        gv_ex = sgn * (-omega * st * spn[i, m])
        v_ex  = -omega * st * sp[i, m]
        if !wall
            check = _weight_check(bs.kb, bf, v_fp, v_ex)
            if check != 0
                k = _boundary_key(s, i, m, check, nq, N)
                key = key == 0 ? k : min(key, k)
                continue
            end
        end
        F1, dv = _side_flux(bs, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
        A[I] = F1
        D[I] = sgn * (af * dv)
        F[I] = -sgn * (c2f * dv)
    end
    return key
end

# Error for the first face point with a non-positive weight over all column blocks.
function _boundary_check(ws, pre, N)
    key = 0
    for k in ws.bfail
        k > 0 && (key = key == 0 ? k : min(key, k))
    end
    key == 0 && return nothing
    s = (key - 1) ÷ (2 * pre.qu1 * N) + 1
    _weight_error((pre.bc.bot, pre.bc.top, pre.bc.lft, pre.bc.rgt)[s], isodd(key) ? 1 : 2)
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
# max(1 - 2βv_q)/min(1 - 2βv_q). Two matrix-free solvers, both to the relative
# residual tolerance pre.mass_tol: Chebyshev iteration (`_cheb_setup!`,
# `_cheb_iterate!`, below) and conjugate gradients started from X = R, the
# exact solution for K = 0 (`_mass_cg_block!`).

# C = A B for small matrices, every entry summed in the order of the inner
# index, four rows of a column at a time in registers (for these sizes faster
# than BLAS also on one thread).
function _mul_loops!(C, A, B)
    m, l = size(A);  n = size(B, 2)
    @inbounds for j in 1:n
        i = 1
        while i + 3 <= m
            s1 = 0.0;  s2 = 0.0;  s3 = 0.0;  s4 = 0.0
            for k in 1:l
                b = B[k, j]
                s1 = muladd(A[i, k], b, s1);      s2 = muladd(A[i + 1, k], b, s2)
                s3 = muladd(A[i + 2, k], b, s3);  s4 = muladd(A[i + 3, k], b, s4)
            end
            C[i, j] = s1;  C[i + 1, j] = s2;  C[i + 2, j] = s3;  C[i + 3, j] = s4
            i += 4
        end
        while i <= m
            s = 0.0
            for k in 1:l
                s = muladd(A[i, k], B[k, j], s)
            end
            C[i, j] = s
            i += 1
        end
    end
    return C
end

# Conjugate gradients on the elements er, one element at a time with the work
# arrays of the block `bk`. Returns the number of systems that did not reach
# the tolerance.
function _mass_cg_block!(X, V, Rv, pre, bk, er, N)
    P = pre.P_h_v;  Pt = pre.P_h_vT;  W = pre.W2d_h;  tol = pre.mass_tol
    nv = pre.qv1;  maxit = nv * nv + 10
    T1 = bk.cg_T1;  T2 = bk.cg_T2;  Y = bk.cg_Y;  G = bk.cg_G
    r = bk.cg_r;  p = bk.cg_p;  Ap = bk.cg_Ap
    applyK!(out, Z) = (_mul_loops!(T1, P, Z); _mul_loops!(Y, T1, Pt); Y .*= G;
                       _mul_loops!(T2, Pt, Y); _mul_loops!(out, T2, P); out)
    nfail = 0
    @inbounds for e in er
        j = (e - 1) % N + 1;  k = (e - 1) ÷ N + 1
        Ve = view(V, :, :, j, k);  Re = view(Rv, :, :, j, k);  Xe = view(X, :, :, j, k)
        _mul_loops!(T1, P, Ve);  _mul_loops!(Y, T1, Pt)         # v_q
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
    return nfail
end

function _cg_report(ws, pre)
    nfail = sum(ws.nfail)
    nfail > 0 && @warn "mass solve: $nfail element systems did not reach the tolerance $(pre.mass_tol)"
    return nothing
end

# The mass systems of all elements by conjugate gradients.
function _mass_solve_cg!(X, V, Rv, pre, ws, N)
    _foreach_block(ws.nb) do b
        ws.nfail[b] = _mass_cg_block!(X, V, Rv, pre, ws.blk[b], ws.eblk[b], N)
    end
    _cg_report(ws, pre)
    return X
end

# Chebyshev iteration (Saad, Iterative Methods for Sparse Linear Systems, 2nd
# ed., Alg. 12.1) for the same systems, on all elements of a block at once. On
# element e the spectrum of M lies in [a₋, a₊], a± = max/min over the q_high
# points of 1 - 2β v_q, which is computed for free; with θ = (a₊ + a₋)/2,
# δ = (a₊ - a₋)/2 and X₀ = 0, n steps give the residual polynomial
# T_n((θ - λ)/δ) / T_n(θ/δ), so
#     ‖R - M X_n‖ ≤ ‖R‖ / cosh(n acosh(θ/δ)),
# and n = ⌈acosh(1/tol) / acosh(θ/δ)⌉ reaches the tolerance without any inner
# product. All elements take the largest n over all blocks (further steps only
# reduce the error of the others, and the result does not depend on the
# blocks); a step costs one application of K and n steps n - 1. The vectors
# are stored as (mode a, element, mode b), so that both contractions of
# P Z P' and of P' H P are single matrix products over the block.
# 1 - 2β v_q ≤ 0 at any point stops the run (the scheme and its analysis need
# 1 - 2βv_h > 0).

# Interval data θ, δ, σ1 = θ/δ and G = 2βv_q W of the elements er, which need
# only v_h. Returns (n, 0) with the largest step count n of the block, or
# (n, e) with e the first element whose mass matrix is not positive definite.
function _cheb_setup!(bk, V, pre, er)
    L = acosh(1.0 / pre.mass_tol)
    Z = bk.ch_Z;  G = bk.ch_G;  θ = bk.ch_th;  δ = bk.ch_de;  σ1 = bk.ch_s1
    _gather!(Z, V, er)
    _contract_fwd!(G, bk.ch_T, pre.P_h_v, pre.P_h_vT, Z)
    G .*= 2.0 .* view(pre.beta_hq, :, er, :)                   # 2βv_q
    n = 1
    @inbounds for e in 1:length(er)
        gmin, gmax = extrema(view(G, :, e, :))
        amin = 1.0 - gmax;  amax = 1.0 - gmin
        amin > 0 || return n, er[e]
        θe = 0.5 * (amax + amin)
        δe = max(0.5 * (amax - amin), 1e-15 * θe)               # δ = 0: one step, X = R/θ
        θ[e] = θe;  δ[e] = δe;  σ1[e] = θe / δe
        n = max(n, ceil(Int, L / acosh(σ1[e])))
    end
    G .*= pre.W_hq                                              # G = 2βv_q W
    return n, 0
end

# The common step count of all blocks after `_cheb_setup!`, or the error for
# the first element whose mass matrix is not positive definite.
function _cheb_steps(ws)
    e = 0
    for f in ws.mfail
        f > 0 && (e = e == 0 ? f : min(e, f))
    end
    e > 0 && error("mass matrix not positive definite on element $e: 1 - 2βv_h ≤ 0")
    return maximum(ws.nsteps)
end

# n Chebyshev steps on the elements er, X = M⁻¹ Rv there.
function _cheb_iterate!(X, Rv, pre, bk, er, n)
    P = pre.P_h_v;  Pt = pre.P_h_vT
    T = bk.ch_T;  S = bk.ch_S;  Y = bk.ch_Y;  G = bk.ch_G
    r = bk.ch_r;  d = bk.ch_d;  Kd = bk.ch_Kd;  Xs = bk.ch_X
    θ = bk.ch_th;  δ = bk.ch_de;  σ1 = bk.ch_s1;  ρ = bk.ch_rho;  ρn = bk.ch_rhon
    _gather!(r, Rv, er)                                         # r₀ = R
    d .= r ./ θ;  fill!(Xs, 0.0);  ρ .= 1.0 ./ σ1
    for step in 1:n
        Xs .+= d
        step == n && break
        _contract_fwd!(Y, T, P, Pt, d);  Y .*= G;  _contract_bwd!(Kd, S, Pt, P, Y)   # Kd = K d
        r .-= d .- Kd                                           # r -= M d
        ρn .= 1.0 ./ (2.0 .* σ1 .- ρ)
        d .= (ρn .* ρ) .* d .+ (2.0 .* ρn ./ δ) .* r
        ρ .= ρn
    end
    _scatter!(X, Xs, er)
    return X
end

# The mass systems of all elements by Chebyshev iteration.
function _mass_solve_cheb!(X, V, Rv, pre, ws, N)
    _foreach_block(ws.nb) do b
        ws.nsteps[b], ws.mfail[b] = _cheb_setup!(ws.blk[b], V, pre, ws.eblk[b])
    end
    n = _cheb_steps(ws)
    _foreach_block(ws.nb) do b
        _cheb_iterate!(X, Rv, pre, ws.blk[b], ws.eblk[b], n)
    end
    return X
end
