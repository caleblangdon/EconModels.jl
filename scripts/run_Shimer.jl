using Pkg; Pkg.activate(@__DIR__)
using EconModels

using BenchmarkTools
using DataFrames
using GLM
using Optim
using Plots
using QuantEcon
using StatsBase
using UnPack

p = ShimerParams(; y_n = 1000)
m = ShimerModel(p)
@time solve!(m; verbose = true)

df = EconModels.simulate(m)

function clean(x; λ = 10^5)
    cyc, trend = hp_filter(log.(x), λ)
    return cyc
end

function table3(m, df = EconModels.simulate(m); cols=[:u, :v, :θ, :f, :p], λ=1e5)
    cleaned_cols = clean.(eachcol(df[:, cols]); λ)
    std_devs = std.(cleaned_cols)
    cleaned_mat = hcat(cleaned_cols...)
    ac = autocor(cleaned_mat, [1]);
    cor_mat = cor(cleaned_mat)
    # Initialize table with variable names
    tab = DataFrame(cor_mat, cols)
    insertcols!(tab, 1, "" => cols)
    pushfirst!(tab, hcat(:ac, ac))
    pushfirst!(tab, hcat(:std, std_devs'))
    return tab
end

bc_plt = scatter(df.u, df.v; legend=false)
title!(bc_plt, "Simulated Beveridge Curve")
display(bc_plt)

function elasticity(m, df = EconModels.simulate(m); verbose = true)
    data = DataFrame(:log_w=>log.(df.w), :log_p=>log.(df.p))
    reg = lm(@formula(log_w ~ log_p), data)
    verbose && println(reg)
    return coef(reg)
end

# std(clean(df.u))
# corr_clean(df, [:u, :v, :θ, :f, :p])
println(table3(m, df))
coefs = elasticity(m, df)


y_n = 100
comp_m = ShimerModel(ShimerParams(; y_n))
solve!(comp_m)
θ = comp_m.θ
f = comp_m.p.f.(θ)
# outer loop
function distance(log_x; θ, f, y_n)
    c = exp(log_x[1])
    tfp = exp(log_x[2])
    p = ShimerParams(; 
        y_n,
        c, 
        M = CobbDouglas([0.72, 0.28]; tfp),
        counter_cyclical_β = true
        )
    m = ShimerModel(p)
    solve!(m)
    dist1 = sum(abs, m.θ .- θ)
    dist2 = sum(abs, m.p.f.(m.θ) .- f)
    return dist1 + dist2
end

res = optimize(log_x -> distance(log_x; θ, f, y_n), [0.95, 1.355], Optim.Options(show_trace=true))
c, tfp = exp.(res.minimizer) # 1.8, 1.6 (?)




p_rigid = ShimerParams(; 
    y_n = 100,
    counter_cyclical_β = true)
m_rigid = ShimerModel(p_rigid)
@time solve!(m_rigid; verbose = true)

df_rigid = EconModels.simulate(m_rigid)
println(table3(m_rigid, df_rigid))
elasticity(m_rigid, df_rigid)


# Benchmarking
# p = ShimerParams()
# m = ShimerModel(p)
# @benchmark solve!(x; verbose=false) setup = (x = deepcopy($m))