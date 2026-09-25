# Orthonormal Jacobi/Legendre polynomials and Gauss–Legendre quadrature
# on the reference interval [-1, 1].

"""
    JacobiP(x, α, β, n)

Orthonormal Jacobi polynomial ``P_n^{(α,β)}`` evaluated at the points `x`
(vector or scalar). With `α = β = 0` these are the L²(-1,1)-orthonormal
Legendre polynomials used as the 1D modal basis.
"""
function JacobiP(x::AbstractVector, alpha::Real, beta::Real, N::Int)
    xp = float.(x)
    PL = zeros(N + 1, length(xp))

    gamma0 = 2^(alpha + beta + 1) / (alpha + beta + 1) *
             gamma(alpha + 1) * gamma(beta + 1) / gamma(alpha + beta + 1)
    PL[1, :] .= 1.0 / sqrt(gamma0)
    N == 0 && return PL[1, :]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[2, :] = ((alpha + beta + 2) .* xp ./ 2 .+ (alpha - beta) / 2) ./ sqrt(gamma1)
    N == 1 && return PL[N + 1, :]

    # three-term forward recurrence
    aold = 2 / (2 + alpha + beta) * sqrt((alpha + 1) * (beta + 1) / (alpha + beta + 3))
    for i in 1:N-1
        h1   = 2i + alpha + beta
        anew = 2 / (h1 + 2) * sqrt((i + 1) * (i + 1 + alpha + beta) *
               (i + 1 + alpha) * (i + 1 + beta) / (h1 + 1) / (h1 + 3))
        bnew = -(alpha^2 - beta^2) / h1 / (h1 + 2)
        PL[i + 2, :] = (1 / anew) .* (-aold .* PL[i, :] .+ (xp .- bnew) .* PL[i + 1, :])
        aold = anew
    end
    return PL[N + 1, :]
end

JacobiP(x::Real, alpha::Real, beta::Real, N::Int) = JacobiP([float(x)], alpha, beta, N)[1]

"""
    GradJacobiP(x, α, β, n)

First derivative of the orthonormal Jacobi polynomial ``P_n^{(α,β)}``.
"""
function GradJacobiP(r::AbstractVector, alpha::Real, beta::Real, N::Int)
    N == 0 && return zeros(length(r))
    return sqrt(N * (N + alpha + beta + 1)) .* JacobiP(r, alpha + 1, beta + 1, N - 1)
end

GradJacobiP(r::Real, alpha::Real, beta::Real, N::Int) = GradJacobiP([float(r)], alpha, beta, N)[1]

"""
    GaussQCofs(q) -> (x, w)

Nodes and weights of the `q`-point Gauss–Legendre rule on [-1, 1]
(Golub–Welsch). Requires `q ≥ 2`.
"""
function GaussQCofs(q::Int)
    kk = collect(1:q-1) .^ 2
    beta_vec = sqrt.(kk ./ (4 .* kk .- 1))
    J   = diagm(1 => beta_vec) + diagm(-1 => beta_vec)
    eig = eigen(J)
    x   = eig.values
    w   = 2 .* eig.vectors[1, :] .^ 2
    idx = sortperm(x)
    return x[idx], w[idx]
end
