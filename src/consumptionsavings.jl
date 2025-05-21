@kwdef struct ConsumptionSavingsParams <: Params
    ρ::Float64 = 0.9
    σ::Float64 = 0.1
    β::Float64 = 0.95
    r::Float64 = 0.04
    γ::Float64 = 2.0
    u::Function = c -> c^(1-γ) / (1-γ)
    u′::Function = c -> c^(-γ)
end

struct ConsumptionSavingsModel <: Model
    p::ConsumptionSavingsParams
    b_grid::Vector{Float64}
    log_y_grid::Vector{Float64}
    Π::Matrix{Float64}
    V::Matrix{Float64}
    c::Matrix{Float64}
    discount::Bool # discount equiv. to paradigm (2), eg. state b is post-interest
end

function ConsumptionSavingsModel(p::Params = ConsumptionSavingsParams(); Nb = 50, Ny = 30, b_up_lim = 100.0, discount = false)
    @unpack ρ, σ, β = p
    mc = rouwenhorst(Ny, ρ, σ)
    Π, log_y_grid = mc.p, mc.state_values
    b_grid = LinRange(0.0, b_up_lim, Nb)
    @tullio V[bx, yx] := (.5*b_grid[bx] + log_y_grid[yx]) / (1-β)
    c = .5*ones(Nb, Ny)
    return ConsumptionSavingsModel(p, b_grid, log_y_grid, Π, V, c, discount)
end

function solve!(m::ConsumptionSavingsModel; verbose = false)
    vfi!(m::ConsumptionSavingsModel; verbose)
end

function vfi!(m::ConsumptionSavingsModel; max_iter=10_000, tol = 1e-8, verbose = false)
    @unpack β, r, u = m.p
    @unpack b_grid, log_y_grid, Π, V, c, discount = m
    b_up_lim = b_grid[end]
    V_new = similar(V)

    for iter in 1:max_iter
        @tullio EV[bx, yx] := V[bx, yx′] * Π[yx, yx′]
        EV_itp = LinearInterpolation((b_grid,log_y_grid), EV)
        for bx in eachindex(b_grid)
            Threads.@threads for yx in eachindex(log_y_grid)
                w = discount ? b_grid[bx] + exp(log_y_grid[yx]) : (1+r)*b_grid[bx] + exp(log_y_grid[yx])
                obj(cs) = discount ? u(cs*w) + β*EV_itp(min((1-cs)*(1+r)*w, b_up_lim), log_y_grid[yx]) : u(cs*w) + β*EV_itp(min((1-cs)*w, b_up_lim), log_y_grid[yx])
                res = optimize(cs -> -obj(cs), 1e-6, 1.0)
                V_new[bx,yx] = -res.minimum
                c[bx,yx] = res.minimizer
            end
        end
        err = abs.((V_new .- V) ./ V)
        max_error = maximum(err)
        V .= V_new
        if verbose && (iter % 100 == 0)
            println("After $iter iterations, largest error is $max_error")
        end
        if max_error < tol
            println("Converged in $iter iterations.")
            break
        elseif iter == max_iter
            println("Failed to converge in $max_iter iterations")
        end
    end
end

function simulate_moments(m::ConsumptionSavingsModel; N = 50_000, T = 1_000)
    @unpack ρ, σ, r = m.p
    @unpack b_grid, log_y_grid, V, c, discount = m
    b_up_lim = b_grid[end]
    b = rand(N) * 1
    mc = rouwenhorst(length(log_y_grid), ρ, σ)
    log_Y = zeros(T,N)
    simulate!(log_Y, mc)
    c_itp = LinearInterpolation((b_grid, log_y_grid), c)
    for tx in 1:T
        Threads.@threads for nx in 1:N
            cs = c_itp(b[nx], log_Y[tx,nx])
            w = discount ? b[nx] + exp(log_Y[tx,nx]) : (1+r)*b[nx] + exp(log_Y[tx,nx])
            b[nx] = discount ? min((1-cs)*(1+r)*w, b_up_lim) : min((1-cs)*w, b_up_lim)
        end
    end
    moments = (;
    mean = mean(b),
    variance = var(b),
    skewness = skewness(b),
    kurtosis = kurtosis(b)
    )
    return moments, b, log_Y[end, :]
end