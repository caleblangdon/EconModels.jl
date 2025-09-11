abstract type DMPParams <: Params end
abstract type DMPModel{P <: DMPParams} <: Model end

using ConcreteStructs
using NLsolve
using QuantEcon
using Roots
using StatsBase
using Tullio

match_from_unemployment(match, θ) = match(1, θ)
match_from_vacancy(match, θ) = match(θ^-1, 1)

@kwdef @concrete struct CobbDouglas
    φ = 0.5
    α = 0.5
end

(match::CobbDouglas)(u, v) = match.φ*u^match.α*v^(1-match.α)
match_from_unemployment(match::CobbDouglas, θ) = match.φ*θ^(1-match.α)
match_from_vacancy(match::CobbDouglas, θ) = match.φ*θ^-match.α


@concrete terse struct SteadyStateDMPParams <: DMPParams
    β
    μ
    z
    b
    δ
    κ
    M
    f
    q
end

function SteadyStateDMPParams(;
    β = 0.995,
    μ = 0.5,
    z = 1.0,
    b = 0.4,
    δ = 0.04,
    κ = 0.5,
    M = CobbDouglas(; φ = 0.5, α = 0.5),
    )

    f(θ) = match_from_unemployment(M, θ)
    q(θ) = match_from_vacancy(M, θ)

    return SteadyStateDMPParams(β, μ, z, b, δ, κ, M, f, q)
end

@concrete terse mutable struct SteadyStateDMPModel{P <: SteadyStateDMPParams} <: DMPModel{P}
    p::P
    u
    w
    θ
    J
end

function SteadyStateDMPModel(p::SteadyStateDMPParams = SteadyStateDMPParams())
    u = 0.0
    w = 0.0
    θ = 1.0
    J = 1.0
    return SteadyStateDMPModel(p, u, w, θ, J)
end

function solve!(m::SteadyStateDMPModel)
    @unpack β, μ, z, b, δ, κ, f, q = m.p
    @unpack u, w, θ, J = m
    F(θ) = κ*(1-β*(1-δ)) - q(θ)*β*((1-μ)*(z-b)-κ*μ*θ)
    # F(θ) = κ - β*q(θ)*((1-μ)*(z-b) - μ*κ*θ + (1-δ)*κ/q(θ))
    θ = find_zero(F, (0.0, 10.0))
    u = δ / (δ + f(θ))
    w = μ*z + (1-μ)*b + μ*κ*θ
    J = (z-w) / (1-β*(1-δ))
    @pack! m = u, w, θ, J
end



@concrete terse struct StochasticDMPParams <: DMPParams
    β
    μ
    b
    δ
    κ
    M
    f
    q

    z_N
    z_grid
    z_Π
end

function StochasticDMPParams(;
    β = 0.995,
    μ = 0.5,
    b = 0.4,
    δ = 0.04,
    κ = 0.5,
    M = CobbDouglas(; φ = 0.5, α = 0.5),
    z_N = 7,
    z_ρ = 0.9,
    z_σ = 0.1,
    )

    f(θ) = match_from_unemployment(M, θ)
    q(θ) = match_from_vacancy(M, θ)

    mc = rouwenhorst(z_N, z_ρ, z_σ)
    z_Π, z_grid = mc.p, exp.(mc.state_values)
    
    return StochasticDMPParams(β, μ, b, δ, κ, M, f, q, z_N, z_grid, z_Π)
end

@concrete terse mutable struct StochasticDMPModel{P <: StochasticDMPParams} <: DMPModel{P}
    p::P
    u # contains steady state u values corresponding to z's
    w
    θ
    J
end

function StochasticDMPModel(p::StochasticDMPParams = StochasticDMPParams();
    θ_guess = 1.0
    )
    @unpack z_N = p
    u = fill(0.0, z_N)
    w = fill(0.0, z_N)
    θ = fill(θ_guess, z_N)
    J = fill(0.0, z_N)

    return StochasticDMPModel(p, u, w, θ, J)
end

function solve!(m::StochasticDMPModel; verbose = false)
    @unpack β, μ, b, δ, κ, f, q, z_N, z_grid, z_Π = m.p
    @unpack u, w, θ, J = m
    function F(θ)
        any(x -> x<0, θ) && return fill(Inf, z_N)
        res = zero(θ)
        for i in eachindex(θ)
            expectation = 0.0
            for j in eachindex(θ)
                expectation += z_Π[i,j]*((1-μ)*(z_grid[j]-b) - μ*κ*θ[j] + (1-δ)*κ/q(θ[j]))
            end
            res[i] = κ - β*q(θ[i])*expectation
        end
        return res
    end
    sol = nlsolve(F, θ; autodiff=:forward, show_trace=verbose)
    verbose && println(sol)
    θ .= sol.zero
    @tullio u[i] = δ / (δ + f(θ[i]))
    @tullio w[i] = μ*z_grid[i] + (1-μ)*b + μ*κ*θ[i]
    @tullio J[i] = z_grid[i] - w[i] + (1-δ)*κ/q(θ[i])
end

function step(m::StochasticDMPModel, u, state_idx)
    @unpack δ, f, z_Π = m.p
    @unpack θ = m
    u = u - f(θ[state_idx])*u + δ*(1-u)
    state_idx = sample(Weights(z_Π[state_idx,:]))
    return u, state_idx
end

function simulate_unemployment(m::StochasticDMPModel, T = 100; burn_in = 100)
    @unpack δ, f, z_N = m.p
    @unpack θ = m
    unemployment_history = Vector{Float64}(undef, T)
    state_idx = div(z_N, 2)
    u = m.u[state_idx]
    for t in 1:burn_in
        u, state_idx = step(m, u, state_idx)
    end
    for t in 1:T
        u, state_idx = step(m, u, state_idx)
        unemployment_history[t] = u
    end
    return unemployment_history
end