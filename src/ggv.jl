struct GGVParams <: Params
    r::Float64                  # weekly interest rate
    δ::Float64                  # exogenous separation
    ψ::Float64                  # total job arrival
    b::Float64                  # benefit flow
    κ::Float64                  # disutility of living apart
    L::Int                      # locations
    θ::Float64                  # probability of local job offer
    σ::Float64                  # wage scale
    μ::Float64                  # wage loc
    Nw::Int                     # number of wage points
    w_grid::Vector{Float64}     # wage grid
    f_grid::Vector{Float64}     # pmf
    F_grid::Vector{Float64}     # cdf
end

function GGVParams(;r = 0.001, δ = 0.0054, ψ = 0.25, b = 0.4, κ = 0.0, L = 9, σ = 0.05, μ = (-(σ^2))/2, Nw = 500)
    θ = 1/L
    lower = 0.0
    upper = exp(μ + 3σ)
    wage_dist = truncated(LogNormal(μ, σ), lower, upper)
    f(w) = pdf(wage_dist, w)
    w_grid = range(lower, upper, Nw)
    f_grid = f.(w_grid)
    f_grid = f_grid ./ sum(f_grid)
    F_grid = cumsum(f_grid)
    return GGVParams(r, δ, ψ, b, κ, L, θ, σ, μ, Nw, w_grid, f_grid, F_grid)
end

mutable struct GGVModel <: Model
    p::GGVParams

    # single agents
    U_s::Float64
    V_s::Vector{Float64}
    w_R_single::Int # single reservation wage index
    
    # couples
    U::Float64
    Ω::Vector{Float64}
    T::Matrix{Float64}
    S::Matrix{Float64}
    w_R_dual::Int # dual searcher reservation wage index
    ŵ_T::Int # double indifference (living together) index
    ŵ_S::Int # double indifference (living apart) index
    φ_i::Vector{Int} # reservation function (inside offers)
    φ_o::Vector{Int} # reservation function (outside offers)
end

function GGVModel(p::GGVParams)
    @unpack r, κ, Nw, b, w_grid = p

    U_s = b / r
    V_s = w_grid ./ r
    w_R_single = 0

    U = (2b) / r
    Ω = (b .+ w_grid) ./ r
    @tullio T[i,j] := (w_grid[i] + w_grid[j]) ./ r
    @tullio S[i,j] := (w_grid[i] + w_grid[j] - κ) ./ r
    w_R_dual = 0
    ŵ_T = 0
    ŵ_S = 0

    φ_i = zeros(Int, Nw)
    φ_o = zeros(Int, Nw)
    return GGVModel(p, U_s, V_s, w_R_single, U, Ω, T, S, w_R_dual, ŵ_T, ŵ_S, φ_i, φ_o)
end

function solve_singles!(m::GGVModel; Δ=2.0, max_iter = 1_000_000, tol=1e-6, verbose = false)
    @unpack r, b, δ, ψ, w_grid, f_grid = parameters(m)
    @unpack U_s, V_s = m
    
    for iter in 1:max_iter
        # unemployed
        EV = ψ * dot(max.(V_s .- U_s, zero(V_s)), f_grid)
        U_s_new = U_s + Δ*(b + EV - r*U_s)

        # employed
        V_s_new = V_s + Δ*(w_grid .+ δ * (U_s .- V_s) - r*V_s)

        U_error = abs(U_s_new - U_s)
        V_error = maximum(abs.(V_s_new .- V_s))
        error = max(U_error, V_error)
        if verbose && (mod(iter, 50_000) == 0)
            println("Iteration $iter")
            println("Max error: $error")
        end
        if error <= tol
            verbose && println("Converged after $iter iterations.")
            break
        end
        U_s = U_s_new
        V_s = V_s_new
    end

    reservation, index = findmin(abs.(V_s .- U_s))
    w_R_single = w_grid[index]
    verbose && println("U_s: $U_s")
    verbose && println("Reservation wage (singles): $w_R_single")
    m.U_s, m.V_s = U_s, V_s
    m.w_R_single = index
end

function solve_dual!(m::GGVModel; Δ=2.0, max_iter = 100_000, tol=1e-6, verbose = false)
    @unpack r, b, δ, ψ, κ, θ, w_grid, f_grid = parameters(m)
    @unpack U, Ω, T, S = m
    U_new = copy(U)
    Ω_new = copy(Ω)
    T_new = copy(T)
    S_new = copy(S)
    
    for iter in 1:max_iter
        # dual searcher
        EV_Ω = ψ * dot(max.(Ω .- U, zero(Ω)), f_grid)
        U_new = U + Δ*(2b + 2 * EV_Ω - r*U)
        
        # searcher-worker
        for i in eachindex(w_grid)
            w1 = w_grid[i]
            EV_T = ψ * θ * dot(max.(T[i, :] .- Ω[i], Ω .- Ω[i], zero(Ω)), f_grid)
            EV_S = ψ * (1 - θ) * dot(max.(S[i, :] .- Ω[i], Ω .- Ω[i], zero(Ω)), f_grid)
            Ω_new[i] = Ω[i] + Δ*(b + w1 + δ * (U - Ω[i]) + EV_T + EV_S - r*Ω[i])
        end
        
        # dual worker (together and separate)
        for i in eachindex(w_grid)
            for j in eachindex(w_grid)
                w1 = w_grid[i]
                w2 = w_grid[j]
                T_new[i, j] = T[i, j] + Δ*(w1 + w2 + δ * (Ω[i] - T[i, j]) + δ * (Ω[j] - T[i, j]) - r*T[i, j])
                S_new[i, j] = S[i, j] + Δ*(w1 + w2 - κ + δ * (Ω[i] - S[i, j]) + δ * (Ω[j] - S[i, j]) - r*S[i, j])
            end
        end
        
        U_error = abs(U_new - U)
        Ω_error = maximum(abs.(Ω_new .- Ω))
        T_error = maximum(abs.(T_new .- T))
        S_error = maximum(abs.(S_new .- S))
        error = max(U_error, Ω_error, T_error, S_error)
        
        if verbose && (mod(iter, 500) == 0)
            println("Iteration $iter")
            println("Max error: $error")
        end
        if error <= tol
            verbose && println("Converged after $iter iterations.")
            break
        end
        U = U_new
        Ω = Ω_new
        T = T_new
        S = S_new
    end

    reservation, index = findmin(abs.(Ω .- U))
    w_reservation_dual = w_grid[index]
    
    verbose && println("U: $U")
    verbose && println("Reservation wage (dual searchers): $w_reservation_dual")
    m.U, m.Ω, m.T, m.S = U, Ω, T, S
    m.w_R_dual = index
end

function solve_double_indifference_points!(m::GGVModel; verbose = true)
    @unpack κ, w_grid, Nw = parameters(m)
    @unpack U, Ω, T, S, w_R_dual = m

    value, m.ŵ_T = findmin([abs(T[i, i] - Ω[i]) for i in eachindex(w_grid)])
    value, m.ŵ_S = findmin([abs(S[i, i] - Ω[i]) for i in eachindex(w_grid)])

    verbose && println("Double-indifference points (ŵ_T): $(w_grid[m.ŵ_T])")
    verbose && println("Double-indifference points (ŵ_S): $(w_grid[m.ŵ_S])")
    
    m.φ_i = max.(w_R_dual, min.([minimum([j for j in 1:Nw if T[i, j] >= Ω[i]]; init=Inf) for i in 1:Nw], 1:Nw))
    m.φ_o = max.(w_R_dual, min.([minimum([j for j in 1:Nw if S[i, j] >= Ω[i]]; init=Inf) for i in 1:Nw], 1:Nw))

    verbose && @show w_grid[m.φ_i[w_R_dual]]

end

function solve!(m::GGVModel; Δ=0.04, verbose = true)
    solve_singles!(m; Δ, verbose)
    solve_dual!(m; Δ, verbose)
    solve_double_indifference_points!(m; verbose)
end

function plot_φs(m::GGVModel)
    @unpack κ, w_grid = parameters(m)
    @unpack φ_o, φ_i = m
    p = plot(w_grid, w_grid[φ_o], label="φ_o(w_1)", xlabel="w_1", ylabel="Reservation Wage", title="Reservation Wage Functions (κ = $κ)")
    plot!(p, w_grid, w_grid[φ_i], label="φ_i(w_1)")
    plot!(p, w_grid[φ_o], w_grid, label="φ_o(w_2)")
    plot!(p, w_grid[φ_i], w_grid, label="φ_i(w_2)")

    xlims!(p, 0.9, 1.1)
    ylims!(p, 0.9, 1.1)
    display(p)
end

function print_reservations(models::Vector{GGVModel})
    nts = []
    for m in models
        nt = (; kappa = m.p.κ,
        single_reservation = m.p.w_grid[m.w_R_single],
        couple_reservation = m.p.w_grid[m.w_R_dual])
        push!(nts, nt)
    end
    df = DataFrame(nts)
    println(df)
end

function print_double_indiff(models::Vector{GGVModel})
    nts = []
    for m in models
        nt = (; kappa = m.p.κ,
        ŵ_T = m.p.w_grid[m.ŵ_T],
        ŵ_S = m.p.w_grid[m.ŵ_S])
        push!(nts, nt)
    end
    df = DataFrame(nts)
    println(df)
end

# p = GGVParams(κ=0.3)
# m = GGVModel(p)
# solve!(m; Δ=2, verbose = true)
# # plot_φs(m)