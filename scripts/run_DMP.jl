using Pkg; Pkg.activate(@__DIR__)
using EconModels

using BenchmarkTools
using Plots
using QuantEcon
using UnPack

p_ss = SteadyStateDMPParams(
    M = EconModels.CobbDouglas(; φ = 0.68, α = 0.5)
    # M = (u,v) -> u*v / ((u^0.5 + v^0.5)^(1/0.5))
)
m_ss = SteadyStateDMPModel(p_ss)
@time solve!(m_ss)
@show m_ss.θ
@show m_ss.p.f(m_ss.θ)
@show unemployment_rate(m_ss)

function unemployment_rate(m)
    δ, f = m.p.δ, m.p.f
    θ = m.θ
    u_rate = δ / (δ + f(θ))
    return u_rate
end

function v(u, m::DMPModel)
    m.p.M isa CobbDouglas || throw(ArgumentError("Requires CobbDouglas matching function"))
    @unpack φ, α = m.p.M
    @unpack δ = m.p
    return ((δ*(1-u))/(φ*u^α))^(1/(1-α))
end

u_range = range(0, 0.25; length=100)
v_range = v.(u_range, Ref(m_ss))
bc_plt = plot(u_range, v_range; label="Beveridge Curve")
θ_range = u_range ./ m_ss.θ
plot!(bc_plt, u_range, θ_range, label="θ")
xlims!(bc_plt, 0, 0.25)
ylims!(bc_plt, 0, 0.25)
display(bc_plt)

p_stoch = StochasticDMPParams(; z_N = 30,
    M = EconModels.CobbDouglas(; φ = 0.68, α = 0.5)
    )
m_stoch = StochasticDMPModel(p_stoch; θ_guess = m_ss.θ)
@time solve!(m_stoch; verbose=false)
@show m_stoch.θ
@show m_stoch.p.f(m_ss.θ)

uh = EconModels.simulate_unemployment(m_stoch, 5_000)
uh_cyclical, uh_trend = hp_filter(uh, 129600)
uh_plt = plot(uh; label="Unemployment")
plot!(uh_plt, uh_trend; label="Unemployment (trend)")
# plot!(uh_plt, uh_cyclical; label="Unemployment (cyclical)")
display(uh_plt)

# raw_volatility = var(log.(uh))
# filtered_volatility = var(log.(uh_trend))
std_dev_cycle = std(uh_cyclical)


# Benchmarking
# p = StochasticDMPParams()
# m = StochasticDMPModel(p)
# @benchmark solve!(x; verbose=false) setup = (x = deepcopy($m))