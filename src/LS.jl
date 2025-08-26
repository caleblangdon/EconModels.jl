using Distributions
using UnPack
using ProgressMeter
using LinearAlgebra
using StatsBase
using Base.Threads

struct LSParams{F <: AbstractFloat, VF <: AbstractVector{F}} <: Params
    β::F
    α::F
    λ::F
    Δ::Int
    ψ_F::Int
    ψ_U::Int
    b::F

    w_N::Int
    w_grid::VF
    f_grid::VF
    F_grid::VF

    s_N::Int
    s_grid::VF

    h_N::Int
    h_grid::VF

    c_grid::VF
    π_grid::VF
end

function LSParams(;
    β = 0.9985,
    α = 0.0009,
    λ = 0.009,
    Δ = 1,
    ψ_F = 30,
    ψ_U = 10,
    b = 0.1,

    μ_w = 0.5,
    σ_w = sqrt(0.1),

    w_low_lim = 0.0,
    w_up_lim = 1.0,
    w_N = 41,
    
    s_low_lim = 0.0,
    s_up_lim = 1.0,
    s_N = 41,
    
    h_low_lim = 1.0,
    h_up_lim = 2.0,
    h_N = 201,

    search_cost_coef = 0.5,
    contact_exp = 0.3,
    )

    w_grid = collect(range(; start=w_low_lim, stop=w_up_lim, length=w_N))

    d = truncated(Normal(μ_w, σ_w), 0.0, 1.0)
    f_grid = pdf.(d, w_grid)
    f_grid ./= sum(f_grid)
    F_grid = cumsum(f_grid)

    s_grid = collect(range(; start=s_low_lim, stop=s_up_lim, length=s_N))
    h_grid = collect(range(; start=h_low_lim, stop=h_up_lim, length=h_N))

    c_grid = search_cost_coef .* s_grid
    π_grid = s_grid .^ contact_exp
    return LSParams(β, α, λ, Δ, ψ_F, ψ_U, b, w_N, w_grid, f_grid, F_grid, s_N, s_grid, h_N, h_grid, c_grid, π_grid)
end

mutable struct LSModel{F <: AbstractFloat, VI <: AbstractVector{Int}, VF <: AbstractVector{F}, MF <: AbstractMatrix{F}} <: Model
    p::LSParams{F, VF}
    τ::F
    U::VF
    optimal_s_idx::VI
    w_R_idx::VI
    W::MF
end

function LSModel(p = LSParams())
    τ = 0.0
    U = fill(p.b / (1-p.β), p.h_N)
    optimal_s_idx = fill(p.s_N, p.h_N)
    w_R_idx = fill(1, p.h_N)
    @tullio W[wx, hx] := (p.w_grid[wx] * p.h_grid[hx]) / (1-p.β)
    return LSModel(p, τ, U, optimal_s_idx, w_R_idx, W)
end

function solve_value_functions!(m::LSModel; tol = 1e-3, max_iter = 10_000, report_steps = 1000, verbose = false)
    @unpack β, α, λ, Δ, ψ_U, ψ_F, b, w_N, w_grid, f_grid, s_N, s_grid, h_grid, h_N, c_grid, π_grid = m.p
    @unpack τ, U, optimal_s_idx, w_R_idx, W = m
    U′ = similar(U)
    W′ = similar(W)
    for iter in 1:max_iter
        @threads for hx in eachindex(h_grid)
            # unemployed
            hx′ = max(hx - ψ_U*Δ, 1)
            EV = dot(f_grid, max.(U[hx′], W[:, hx′]))
            U′[hx], optimal_s_idx[hx] = findmax((1-τ)*b .- c_grid + β*(1-α)*((1.0 .- π_grid)*U[hx′] + π_grid*EV))
            
            # employed
            for wx in eachindex(w_grid)
                W′[wx, hx] = (1-τ)*w_grid[wx]*h_grid[hx] + β*(1-α)*(λ*U[max(hx-Δ*ψ_F, 1)] + (1-λ)*W[wx, min(hx+Δ, h_N)])
            end
        end
        U_error = maximum(abs, U′ .- U)
        W_error = maximum(abs, W′ .- W)
        worst_error = max(U_error, W_error)
        U .= U′
        W .= W′
        
        if worst_error < tol
            verbose && println("   Converged after $iter iterations with maximum error $worst_error.")
            break
        elseif verbose && (iter % report_steps == 0)
            println("   After $iter iterations, max error is $worst_error")
        elseif iter == max_iter
            println("   Failed to converge after $max_iter iterations.")
        end
    end
    for hx in eachindex(h_grid)
        res = findfirst(x -> W[x, hx] >= U[hx], 1:w_N)
        w_R_idx[hx] = !isnothing(res) ? res : 1
    end
end

@kwdef mutable struct LSWorker
    id::Int = 1
    alive::Bool = true
    employed::Bool = false
    w_idx::Int = 0 # 0 is unemployed
    h_idx::Int = 1
end

function step!(m::LSModel, worker::LSWorker, shocks, t)
    @unpack β, α, λ, Δ, ψ_U, ψ_F, w_N, w_grid, f_grid, s_N, s_grid, h_N, h_grid, π_grid = m.p
    @unpack optimal_s_idx, w_R_idx = m
    @unpack id, alive, employed, w_idx, h_idx = worker
    @unpack death, arrival, separation, offer_idxs = shocks

    if !worker.alive || death[id, t] < α
        worker.alive = false
    else
        if !worker.employed
            worker.h_idx = max(h_idx - ψ_U*Δ, 1)
            if arrival[id, t] < π_grid[optimal_s_idx[worker.h_idx]]
                offer_idx = offer_idxs[id, t]
                if offer_idx >= w_R_idx[worker.h_idx]
                    worker.employed = true
                    worker.w_idx = offer_idx
                end
            end
        else
            if separation[id, t] < λ
                worker.h_idx = max(h_idx-Δ*ψ_F, 1)
                worker.employed = false
                worker.w_idx = 0
            else
                worker.h_idx = min(h_idx+Δ, h_N)
            end
        end
    end
end

function pay_taxes(m::LSModel, worker::LSWorker)
    if !worker.alive
        return missing
    else
        if worker.employed
            return m.τ * m.p.w_grid[worker.w_idx] * m.p.h_grid[worker.h_idx]
        else
            return (m.τ - 1) * m.p.b
        end
    end
end

struct Shocks{MI <: AbstractMatrix{<:Integer}, MF <: AbstractMatrix{<:AbstractFloat}}
    death::MF
    arrival::MF
    separation::MF
    offer_idxs::MI
end

function Shocks(m::LSModel, N = 10_000, T = 100; burn_in = 100)
    death_shocks = rand(N, T + burn_in)
    arrival_shocks = rand(N, T + burn_in)
    separation_shocks = rand(N, T + burn_in)
    offer_idxs = sample(1:m.p.w_N, Weights(m.p.f_grid), (N, T + burn_in))
    return Shocks(death_shocks, arrival_shocks, separation_shocks, offer_idxs)
end

function simulate_budget_surplus(m::LSModel, shocks, N, T; burn_in)
    workers = [LSWorker(id=n) for n in 1:N]
    tax_payments = Matrix{Union{Float64,Missing}}(undef, N, T)
    employment_status = Matrix{Union{Bool,Missing}}(undef, N, T)
    for t in 1:burn_in
        Threads.@threads for worker in workers
            step!(m, worker, shocks, t)
        end
    end
    for t in 1:T
        Threads.@threads for worker in workers
            step!(m, worker, shocks, t)
            tax_payments[worker.id, t] = pay_taxes(m, worker)
            employment_status[worker.id, t] = worker.alive ? worker.employed : missing
        end
    end
    return mean(skipmissing(tax_payments)), 1 - mean(skipmissing(employment_status[:,T]))
end

function solve!(m::LSModel;
    N = 10_000, T = 100, burn_in = 100,
    τ_low = 0.0, τ_high = 1.0, tax_tol = 1e-8, vfi_tol = 1e-3, max_iter = 10_000, report_steps = 500, verbose = true)
    shocks = Shocks(m, N, T; burn_in)
    for iter in 1:max_iter
        m.τ = (τ_low + τ_high) / 2
        solve_value_functions!(m; tol = vfi_tol, max_iter, report_steps)
        surplus, u_rate = simulate_budget_surplus(m, shocks, N, T; burn_in)

        if verbose
            println("Current tax rate $(m.τ) results in budget surplus $surplus, after $iter iterations. Unemployment rate is $u_rate.")
        end

        if surplus < 0.0
            τ_low = m.τ
        else
            τ_high = m.τ
        end

        if abs(surplus) < tax_tol
            println("Converged to tax rate $(m.τ), which results in budget surplus $surplus, after $iter iterations. Unemployment rate is $u_rate.")
            break
        elseif iter == max_iter
            println("Failed to converge!")
        end
    end
    return 
end