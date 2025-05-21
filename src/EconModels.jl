module EconModels

using DataFrames
using Distributions
using Interpolations
using Optim
using Plots
using ProgressMeter
using QuantEcon
using Statistics
using Tullio
using UnPack
using LinearAlgebra
using Random

abstract type Params end
abstract type Model end
abstract type EstimationProcedure end

# Example Usage (for model "Type"):
# p = Params<modelname>()
# m = Model<modelname>(p)
# solve!(m)

parameters(m::Model) = m.p


include("ggv.jl")
export GGVParams, GGVModel

include("consumptionsavings.jl")
export ConsumptionSavingsParams, ConsumptionSavingsModel

function test_func()
    println("Test 9")
end
export test_func

export solve!, simulate_moments

end
