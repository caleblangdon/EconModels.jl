using EconModels
using Statistics
using Plots

discount = false

p = ConsumptionSavingsParams()
m = ConsumptionSavingsModel(p; Nb=300, Ny=100, discount)
solve!(m)
moments, b, log_y = simulate_moments(m; T=1_001, N=50_000)
pl = plot(sort(b), (1:length(b)) ./ length(b), label="Asset CDF", title="Caleb's Code (Paradigm $(discount ? 2 : 1))")
display(pl)
println("== Moments (Paradigm $(discount ? 2 : 1)) ==")
println("Mean: $(moments.mean)")
println("Variance: $(moments.variance)")
println("Skewness: $(moments.skewness)")
println("Kurtosis: $(moments.kurtosis)")