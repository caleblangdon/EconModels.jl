using Pkg; Pkg.activate(@__DIR__)
using EconModels

using BenchmarkTools
using Plots
using QuantEcon
using UnPack

p_ss = SteadyStateDMPParams()
m_ss = SteadyStateDMPModel(p_ss)
@time solve!(m_ss)
@show m_ss.θ

# function v(u, m::DMPModel)
#     @unpack δ, φ, α = m.p
#     return ((δ*(1-u))/(φ*u^α))^(1/(1-α))
# end

# u_range = range(0, 0.25; length=100)
# v_range = v.(u_range, Ref(m_ss))
# plt_ss = plot(u_range, v_range; label="Beveridge Curve")
# θ_range = u_range ./ m_ss.θ
# plot!(plt_ss, u_range, θ_range)
# xlims!(plt_ss, 0, 0.25)
# ylims!(plt_ss, 0, 0.25)

p_stoch = StochasticDMPParams(; z_N = 30)
m_stoch = StochasticDMPModel(p_stoch; θ_guess = m_ss.θ)
@time solve!(m_stoch; verbose=false)
@show m_stoch.θ

uh = EconModels.simulate_unemployment(m_stoch, 100)
uh_cyclical, uh_trend = hp_filter(uh, 1600)
plt = plot(uh)
plot!(plt, uh_trend)
display(plt)

raw_volatility = var(log.(uh))
filtered_volatility = var(log.(uh_trend))


# Benchmarking
# p = StochasticDMPParams()
# m = StochasticDMPModel(p)
# @benchmark solve!(x; verbose=false) setup = (x = deepcopy($m))