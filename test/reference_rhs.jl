# Reference implementation used by the tests: the element-by-element RHS of
# WesterveltEDG.jl v0.1.0 (commit 4c9ec1d) with the dense LU mass solve,
# kept verbatim except for the function names. The boundary kinds of v0.3.0
# reach it through equivalent parameters (see `reference_pre` in runtests.jl).
# The production RHS in src/rhs.jl must agree with it to rounding error.

# Semi-discrete right-hand side of the EDG scheme for the Westervelt equation
# in first-order-in-time form  u_t = v,
#     (1 - 2βv) v_t = ∇·(c²∇u) + ∇·(α∇v) + f,
# on a uniform Cartesian mesh of the unit square with tensor-product Legendre
# bases of degree q_u (for u) and q_v (for v).
#
# Layout: u[j, k, :, :] holds the (q_u+1)×(q_u+1) modal coefficients on the
# element in row j (y-direction) and column k (x-direction); likewise v.
#
# The displacement is advanced as ũ = u - u₀ with u₀ the initial profile
# (see `solve`), which only adds the known terms ∇·(c²∇u₀) to the source and
# -(c²∇u₀)·n to the boundary data.
#
# All coefficient and data arrays are read from the precomputed NamedTuple
# `pre` built in solver.jl.

# Pointwise absorbing-boundary flux on a physical boundary face.
# Inputs are values at the face quadrature points; `*_ex` are the boundary
# data. Returns (F1, dv): the normal-flux term entering the v-equation and
# the v-trace correction v* - v_h (scaled by 1/(c √(1-βv_h))).
@inline function _abc_face_reference(pre, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)
    gamma_abc = pre.gamma_abc;  eta_abc = pre.eta_abc;  sigma = pre.sigma

    rho_num = eta_abc .* (c2f .* grad_u_n .+ af .* grad_v_n) .+
              gamma_abc .* cf .* sqrt.(1.0 .- bf .* v_fp) .* v_fp
    rho_ex  = eta_abc .* (c2f .* gu_ex .+ af .* gv_ex) .+
              gamma_abc .* cf .* sqrt.(1.0 .- bf .* v_ex) .* v_ex
    rho_net = rho_num .- rho_ex

    sqf = sqrt.(1.0 .- bf .* v_fp)
    F1  = c2f .* grad_u_n .+ af .* grad_v_n .- (eta_abc + sigma * gamma_abc) .* rho_net
    dv  = -(gamma_abc - sigma * eta_abc) ./ (cf .* sqf) .* rho_net
    return F1, dv
end

function edg_rhs_reference(u::Array{Float64,4}, v::Array{Float64,4},
                 t::Float64, N::Int, h::Float64, pre::NamedTuple)

    qu1 = pre.qu1;  qv1 = pre.qv1;  omega = pre.omega
    w   = pre.w
    P_u = pre.P_u;  P_v = pre.P_v
    Ebl_u = pre.Ebl_u;  Ebr_u = pre.Ebr_u
    E1bl_u = pre.E1bl_u; E1br_u = pre.E1br_u
    Ebl_v = pre.Ebl_v;  Ebr_v = pre.Ebr_v
    E1bl_v = pre.E1bl_v; E1br_v = pre.E1br_v
    W2d_h = pre.W2d_h
    P_h_v = pre.P_h_v;  Phi2d_h_v = pre.Phi2d_h_v

    # interelement flux parameters (θ, τ, λ_u, λ_v)
    theta    = pre.theta
    tau      = pre.tau
    lambda_u = pre.lambda_u
    lambda_v = pre.lambda_v

    two_h   = 2.0 / h
    four_hh = 4.0 / (h * h)

    # time factors of the manufactured solution u = cos(ωt) Φ(x); both are
    # switched off outside the manufactured-solution mode, which leaves only
    # the u₀ terms of the ũ = u - u₀ change of variables.
    ct = pre.mms_on ? cos(omega * t) : 0.0
    st = pre.mms_on ? sin(omega * t) : 0.0

    ut = zeros(size(u))
    vt = zeros(size(v))
    vt_rhs_all = zeros(N, N, qv1, qv1)   # v-residual before the nonlinear mass solve

    for j in 1:N
        for k in 1:N

            uloc = u[j, k, :, :]
            vloc = v[j, k, :, :]

            # ---- volume stiffness ----
            vt_rhs = -four_hh * reshape(
                         pre.Mc2_uv[j,k] * vec(uloc) + pre.Ma_vv[j,k] * vec(vloc),
                         qv1, qv1)

            # ---- manufactured source (+ ∇·(c²∇u₀) from the ũ shift) ----
            SP   = pre.mms_SP[j, k, :, :]
            Cdiv = pre.mms_Cdiv[j, k, :, :]
            Ddiv = pre.mms_Ddiv[j, k, :, :]
            BF   = pre.mms_BF[j, k, :, :]

            v_h_mms  = -omega * st * SP
            vt_h_mms = -omega^2 * ct * SP
            fv_h     = (1.0 .- 2.0 .* BF .* v_h_mms) .* vt_h_mms .-
                       ct .* Cdiv .+ omega * st .* Ddiv .+ Cdiv

            vt_rhs = vt_rhs .+ P_h_v' * (W2d_h .* fv_h) * P_h_v

            # ---- localized source (source mode) ----
            if pre.src_on
                vt_rhs = vt_rhs .+ pre.src_time(t) .* pre.src_proj[j, k, :, :]
            end

            f = zeros(qu1, qu1)

            # ==================================================================
            #  BOTTOM FACE  (n = (0,-1))
            # ==================================================================
            if j == 1
                c2f = pre.bot_c2[j, k, :]
                af  = pre.bot_af[j, k, :]
                cf  = pre.bot_cf[j, k, :]
                bf  = pre.bot_bf[j, k, :]

                v_fp  = P_v * (Ebl_v * vloc)'
                uy_fp = P_u * (two_h * (E1bl_u * uloc))'
                vy_fp = P_v * (two_h * (E1bl_v * vloc))'
                grad_u_n = -uy_fp
                grad_v_n = -vy_fp

                sp_bc  = pre.bc_bot_sp[k, :]
                spy_bc = pre.bc_bot_spy[k, :]
                gu_ex =  -ct .* spy_bc .+ spy_bc        # minus ∇u₀·n = -spy_bc
                gv_ex =   omega * st .* spy_bc
                v_ex  =  -omega * st .* sp_bc

                F1, dv_pts = _abc_face_reference(pre, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)

                vt_rhs = vt_rhs .+
                         two_h  .* Ebl_v'  * (P_v' * (w .* F1))' .-
                         four_hh .* E1bl_v' * (P_v' * (w .* (af .* dv_pts)))'
                f = f .+ E1bl_u' * (P_u' * (w .* (c2f .* dv_pts)))'

            else
                uout = u[j-1, k, :, :]
                vout = v[j-1, k, :, :]
                c2f = pre.bot_c2[j, k, :]
                af  = pre.bot_af[j, k, :]

                v_jump_p = P_v * (Ebr_v * vout .- Ebl_v * vloc)'
                jgu_p    = P_u * (two_h .* (E1br_u * uout .- E1bl_u * uloc))'
                jgv_p    = P_v * (two_h .* (E1br_v * vout .- E1bl_v * vloc))'
                gu_star_p = P_u * (theta .* E1bl_u * uloc .+ (1-theta) .* E1br_u * uout)'
                gv_star_p = P_v * (theta .* E1bl_v * vloc .+ (1-theta) .* E1br_v * vout)'

                dv_p   = theta .* v_jump_p .- tau .* (c2f .* jgu_p .+ af .* jgv_p)
                flux_p = c2f .* gu_star_p .+ af .* gv_star_p
                pen_p  = (c2f .* lambda_u .+ af .* lambda_v) .* v_jump_p

                f = f .+ E1bl_u' * (P_u' * (w .* (c2f .* dv_p)))'
                vt_rhs = vt_rhs .-
                         four_hh .* Ebl_v'  * (P_v' * (w .* flux_p))' .-
                         four_hh .* E1bl_v' * (P_v' * (w .* (af .* dv_p)))' .+
                         two_h   .* Ebl_v'  * (P_v' * (w .* pen_p))'
            end

            # ==================================================================
            #  TOP FACE  (n = (0,+1))
            # ==================================================================
            if j == N
                c2f = pre.top_c2[j, k, :]
                af  = pre.top_af[j, k, :]
                cf  = pre.top_cf[j, k, :]
                bf  = pre.top_bf[j, k, :]

                v_fp  = P_v * (Ebr_v * vloc)'
                uy_fp = P_u * (two_h * (E1br_u * uloc))'
                vy_fp = P_v * (two_h * (E1br_v * vloc))'
                grad_u_n = uy_fp
                grad_v_n = vy_fp

                sp_bc  = pre.bc_top_sp[k, :]
                spy_bc = pre.bc_top_spy[k, :]
                gu_ex =  ct .* spy_bc .- spy_bc         # minus ∇u₀·n = +spy_bc
                gv_ex = -omega * st .* spy_bc
                v_ex  = -omega * st .* sp_bc

                F1, dv_pts = _abc_face_reference(pre, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)

                vt_rhs = vt_rhs .+
                         two_h   .* Ebr_v'  * (P_v' * (w .* F1))' .+
                         four_hh .* E1br_v' * (P_v' * (w .* (af .* dv_pts)))'
                f = f .- E1br_u' * (P_u' * (w .* (c2f .* dv_pts)))'

            else
                uout = u[j+1, k, :, :]
                vout = v[j+1, k, :, :]
                c2f = pre.top_c2[j, k, :]
                af  = pre.top_af[j, k, :]

                v_jump_p = P_v * (Ebl_v * vout .- Ebr_v * vloc)'
                jgu_p    = P_u * (two_h .* (.-E1bl_u * uout .+ E1br_u * uloc))'
                jgv_p    = P_v * (two_h .* (.-E1bl_v * vout .+ E1br_v * vloc))'
                gu_star_p = P_u * ((1-theta) .* E1br_u * uloc .+ theta .* E1bl_u * uout)'
                gv_star_p = P_v * ((1-theta) .* E1br_v * vloc .+ theta .* E1bl_v * vout)'

                dv_p   = (1-theta) .* v_jump_p .- tau .* (c2f .* jgu_p .+ af .* jgv_p)
                flux_p = c2f .* gu_star_p .+ af .* gv_star_p
                pen_p  = (c2f .* lambda_u .+ af .* lambda_v) .* v_jump_p

                f = f .- E1br_u' * (P_u' * (w .* (c2f .* dv_p)))'
                vt_rhs = vt_rhs .+
                         four_hh .* Ebr_v'  * (P_v' * (w .* flux_p))' .+
                         four_hh .* E1br_v' * (P_v' * (w .* (af .* dv_p)))' .+
                         two_h   .* Ebr_v'  * (P_v' * (w .* pen_p))'
            end

            # ==================================================================
            #  LEFT FACE  (n = (-1,0)); the y-face pattern applied to transposes
            # ==================================================================
            if k == 1
                c2f = pre.lft_c2[j, k, :]
                af  = pre.lft_af[j, k, :]
                cf  = pre.lft_cf[j, k, :]
                bf  = pre.lft_bf[j, k, :]

                uloc_T = uloc';  vloc_T = vloc'
                v_fp  = P_v * (Ebl_v * vloc_T)'
                ux_fp = P_u * (two_h * (E1bl_u * uloc_T))'
                vx_fp = P_v * (two_h * (E1bl_v * vloc_T))'
                grad_u_n = -ux_fp
                grad_v_n = -vx_fp

                sp_bc  = pre.bc_lft_sp[j, :]
                spx_bc = pre.bc_lft_spx[j, :]
                gu_ex =  -ct .* spx_bc .+ spx_bc        # minus ∇u₀·n = -spx_bc
                gv_ex =   omega * st .* spx_bc
                v_ex  =  -omega * st .* sp_bc

                F1, dv_pts = _abc_face_reference(pre, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)

                vt_abc = two_h   .* Ebl_v'  * (P_v' * (w .* F1))' .-
                         four_hh .* E1bl_v' * (P_v' * (w .* (af .* dv_pts)))'
                f_abc  = E1bl_u' * (P_u' * (w .* (c2f .* dv_pts)))'

                vt_rhs = vt_rhs .+ vt_abc'
                f      = f      .+ f_abc'

            else
                uout = u[j, k-1, :, :]'
                vout = v[j, k-1, :, :]'
                c2f = pre.lft_c2[j, k, :]
                af  = pre.lft_af[j, k, :]

                v_jump_p = P_v * (Ebr_v * vout .- Ebl_v * vloc')'
                jgu_p    = P_u * (two_h .* (E1br_u * uout .- E1bl_u * uloc'))'
                jgv_p    = P_v * (two_h .* (E1br_v * vout .- E1bl_v * vloc'))'
                gu_star_p = P_u * (theta .* E1bl_u * uloc' .+ (1-theta) .* E1br_u * uout)'
                gv_star_p = P_v * (theta .* E1bl_v * vloc' .+ (1-theta) .* E1br_v * vout)'

                dv_p   = theta .* v_jump_p .- tau .* (c2f .* jgu_p .+ af .* jgv_p)
                flux_p = c2f .* gu_star_p .+ af .* gv_star_p
                pen_p  = (c2f .* lambda_u .+ af .* lambda_v) .* v_jump_p

                f      = f      .+ (P_u' * (w .* (c2f .* dv_p))) * E1bl_u
                vt_rhs = vt_rhs .-
                         four_hh .* (P_v' * (w .* flux_p)) * Ebl_v .-
                         four_hh .* (P_v' * (w .* (af .* dv_p))) * E1bl_v .+
                         two_h   .* (P_v' * (w .* pen_p)) * Ebl_v
            end

            # ==================================================================
            #  RIGHT FACE  (n = (+1,0)); the y-face pattern applied to transposes
            # ==================================================================
            if k == N
                c2f = pre.rgt_c2[j, k, :]
                af  = pre.rgt_af[j, k, :]
                cf  = pre.rgt_cf[j, k, :]
                bf  = pre.rgt_bf[j, k, :]

                uloc_T = uloc';  vloc_T = vloc'
                v_fp  = P_v * (Ebr_v * vloc_T)'
                ux_fp = P_u * (two_h * (E1br_u * uloc_T))'
                vx_fp = P_v * (two_h * (E1br_v * vloc_T))'
                grad_u_n = ux_fp
                grad_v_n = vx_fp

                sp_bc  = pre.bc_rgt_sp[j, :]
                spx_bc = pre.bc_rgt_spx[j, :]
                gu_ex =  ct .* spx_bc .- spx_bc         # minus ∇u₀·n = +spx_bc
                gv_ex = -omega * st .* spx_bc
                v_ex  = -omega * st .* sp_bc

                F1, dv_pts = _abc_face_reference(pre, c2f, af, cf, bf, v_fp, grad_u_n, grad_v_n, gu_ex, gv_ex, v_ex)

                vt_abc = two_h   .* Ebr_v'  * (P_v' * (w .* F1))' .+
                         four_hh .* E1br_v' * (P_v' * (w .* (af .* dv_pts)))'
                f_abc  = .-E1br_u' * (P_u' * (w .* (c2f .* dv_pts)))'

                vt_rhs = vt_rhs .+ vt_abc'
                f      = f      .+ f_abc'

            else
                uout = u[j, k+1, :, :]'
                vout = v[j, k+1, :, :]'
                c2f = pre.rgt_c2[j, k, :]
                af  = pre.rgt_af[j, k, :]

                v_jump_p = P_v * (Ebl_v * vout .- Ebr_v * vloc')'
                jgu_p    = P_u * (two_h .* (.-E1bl_u * uout .+ E1br_u * uloc'))'
                jgv_p    = P_v * (two_h .* (.-E1bl_v * vout .+ E1br_v * vloc'))'
                gu_star_p = P_u * ((1-theta) .* E1br_u * uloc' .+ theta .* E1bl_u * uout)'
                gv_star_p = P_v * ((1-theta) .* E1br_v * vloc' .+ theta .* E1bl_v * vout)'

                dv_p   = (1-theta) .* v_jump_p .- tau .* (c2f .* jgu_p .+ af .* jgv_p)
                flux_p = c2f .* gu_star_p .+ af .* gv_star_p
                pen_p  = (c2f .* lambda_u .+ af .* lambda_v) .* v_jump_p

                f      = f      .- (P_u' * (w .* (c2f .* dv_p))) * E1br_u
                vt_rhs = vt_rhs .+
                         four_hh .* (P_v' * (w .* flux_p)) * Ebr_v .+
                         four_hh .* (P_v' * (w .* (af .* dv_p))) * E1br_v .+
                         two_h   .* (P_v' * (w .* pen_p)) * Ebr_v
            end

            vt_rhs_all[j, k, :, :] = vt_rhs

            # ---- displacement equation (u-space, pinned c²-stiffness, Cholesky) ----
            rhs_ut      = pre.Mc2_uv[j,k]' * vec(vloc) .- vec(f)
            rhs_ut[1]  += vloc[1, 1]
            R            = pre.R_ut[j, k]
            ut[j, k, :, :] = reshape(R \ (R' \ rhs_ut), qu1, qu1)

        end
    end

    # ---- nonlinear mass (1 - 2βv_h) solve per element (v-space) ----
    for j in 1:N, k in 1:N
        vloc = v[j, k, :, :]
        BF   = pre.mms_BF[j, k, :, :]
        v_quad_h = P_h_v * vloc * P_h_v'
        wv       = vec((1.0 .- 2.0 .* BF .* v_quad_h) .* W2d_h)
        Mw_h     = Phi2d_h_v' * (wv .* Phi2d_h_v)
        vt[j, k, :, :] = reshape(Mw_h \ vec(vt_rhs_all[j, k, :, :]), qv1, qv1)
    end

    return ut, vt
end
