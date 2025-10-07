using ConcreteStructs
using DataFrames
using Distributions
using NLsolve
using SparseArrays
using StatsBase
using Tullio
using UnPack

@concrete terse struct ShimerParams <: Params
    r
    z
    M
    f
    q
    β_grid
    c
    λ
    
    Δ
    γ
    σ
    y_N
    y_grid
    y_Π
    
    p_grid
    s_grid
end

function ShimerParams(;
    p_star = 1.0,
    s_star = 0.1, 
    r = 0.012,
    z = 0.4,
    M = CobbDouglas([0.72, 0.28]; tfp = 1.355),
    β_star = 0.72,
    c = 0.213,
    σ = 0.0165,
    γ = 0.004,
    y_n = 100,
    counter_cyclical_β = false
    )

    f(θ) = match_from_unemployment(M, θ)
    q(θ) = match_from_vacancy(M, θ)

    λ = γ*y_n
    Δ = σ / sqrt(λ)
    y_N = 2*y_n + 1
    y_grid = LinRange(-y_n*Δ, y_n*Δ, y_N)
    y_Π = spzeros(y_N, y_N)
    y_Π[1, 2] = 1.0
    y_Π[y_N, y_N-1] = 1.0
    for y_idx in 2:y_N-1
        p = 0.5 * (1.0 - y_grid[y_idx] / (y_n * Δ))
        y_Π[y_idx, y_idx+1] = p
        y_Π[y_idx, y_idx-1] = 1 - p
    end

    p_grid = z .+ exp.(y_grid) * (p_star - z)
    s_grid = fill(s_star, y_N) # exp.(y_grid) * s_star
    β_grid = counter_cyclical_β ? 1.0 ./ (1 .+ exp.( .- ( .- y_grid .+ log(β_star/(1-β_star))))) : fill(β_star, y_N)

    return ShimerParams(r, z, M, f, q, β_grid, c, λ, Δ, γ, σ, y_N, y_grid, y_Π, p_grid, s_grid)
end

@concrete terse struct ShimerModel{P <: ShimerParams} <: Model
    p::P
    u
    w
    θ
    J
end

function ShimerModel(p::ShimerParams = ShimerParams();
    θ_guess = 1.0)
    @unpack y_N = p
    u = fill(0.0, y_N)
    w = fill(0.0, y_N)
    θ = fill(θ_guess, y_N)
    J = fill(0.0, y_N)

    return ShimerModel(p, u, w, θ, J)
end

function solve!(m::ShimerModel; verbose = false)
    @unpack β_grid, r, s_grid, λ, f, q, p_grid, z, c, y_N, y_Π = m.p
    @unpack u, w, θ, J = m
    
    function F(log_θ)
        interior_θ = exp.(log_θ)
        qθ = q.(interior_θ)
        v = 1 ./ qθ
        expectation = y_Π * v
        res = ((r .+ s_grid .+ λ) ./ qθ) .+ β_grid .* interior_θ .- (1 .- β_grid) .* ((p_grid .- z) ./ c) .- λ*expectation
        return res
    end
    sol = nlsolve(F, log.(θ); autodiff=:forward, show_trace=verbose)
    θ .= exp.(sol.zero)
    u .= s_grid ./ (s_grid .+ f.(θ))
    w .= β_grid .* p_grid .+ (1 .- β_grid)*z .+ β_grid*c .* θ
    J .= p_grid .- w .+ (1 .- s_grid)*c ./ q.(θ)
end

function step(m::ShimerModel, u, state_idx, duration)
    @unpack s_grid, f, y_Π = m.p
    @unpack θ = m
    state_idx = sample(Weights(y_Π[state_idx,:]))
    u = u + duration * (s_grid[state_idx]*(1-u) - f(θ[state_idx])*u)
    return u, state_idx
end

function simulate(m::ShimerModel, T = 212; burn_in = 1_000)
    @unpack f, y_N, p_grid, y_grid, λ = m.p
    @unpack θ = m
    state_idx = div(y_N, 2)
    u = m.u[state_idx]
    cont_df = DataFrame(
        t = Float64[],
        u = Float64[],
        state_idx = Int[],
    )
    shock_dist = Exponential(1/λ)
    t = -burn_in

    while t <= T
        time_passed = rand(shock_dist)
        u, state_idx = step(m, u, state_idx, time_passed)
        push!(cont_df, (; t, u, state_idx))
        t += time_passed
    end

    df = DataFrame(
        quarter = 1:T,
        state_idx = Vector{Int}(undef, T),
        u = Vector{Float64}(undef, T)
        )
    
    for q in 1:T
        idx = findlast(<=(q), cont_df.t)
        df.u[q] = cont_df.u[idx]
        df.state_idx[q] = cont_df.state_idx[idx]
    end

    df.θ = θ[df.state_idx]
    df.y = y_grid[df.state_idx]
    df.v = df.θ .* df.u
    df.f = f.(df.θ)
    df.p = p_grid[df.state_idx]
    df.w = m.w[df.state_idx]

    return df
end