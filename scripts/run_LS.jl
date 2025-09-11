using Pkg
Pkg.activate(@__DIR__)
using EconModels
using Plots
using BenchmarkTools

p = LSParams()
m = LSModel(p)

EconModels.solve!(m;
    N = 10_000, T = 100, burn_in = 100,
    τ_guess = 0.1,
    tax_tol = 1e-9,
    vfi_tol = 1e-3,
    max_iter = 10_000,
    report_steps = 500,
    verbose = true
    )

# plot(m.U)
# plot(m.optimal_s_idx)
# plot(m.w_R_idx)
# plot(m.W[20,:])
# plot(m.W[:,100])

# using PlotlyJS
# x = m.p.w_grid
# y = m.p.h_grid
# z = m.W

# plt = PlotlyJS.plot(
#     PlotlyJS.surface(x=x, y=y, z=z),
#     Layout(
#             scene = attr(
#                 xaxis = attr(title="Wages"),
#                 yaxis = attr(title="Human Capital"),
#                 zaxis = attr(title="Value Function W"),
#             ),
#             title = "Employed Value Function Surface"
#         )
# )

# display(plt)

# Benchmarking
# p = LSParams()
# m = LSModel(p)
# time = @belapsed solve!(x; τ_guess=0.1, verbose=false) setup = (x = deepcopy($m))