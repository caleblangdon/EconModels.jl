module EconModels

using ConcreteStructs
using DataFrames
using Distributions
using ForwardDiff
using Interpolations
using Optim
using Plots
using ProgressMeter
using QuantEcon
using Statistics
using StatsBase
using Tullio
using UnPack
using LinearAlgebra
using NLsolve
using Random
using Roots

abstract type Params end
abstract type Model end
abstract type EstimationProcedure end

# Example Usage:
# p = Params<modelname>()
# m = Model<modelname>(p)
# solve!(m)

parameters(m::Model) = m.p
include("common_functions.jl")
export CobbDouglas, CES, DHRW, CRRA, marginal


include("GGV.jl")
export GGVParams, GGVModel

include("consumptionsavings.jl")
export ConsumptionSavingsParams, ConsumptionSavingsModel

include("LS.jl")
export LSParams, LSModel

export DMPParams, DMPModel
export SteadyStateDMPParams, SteadyStateDMPModel, StochasticDMPParams, StochasticDMPModel
include("DMP.jl")

function test_func()
    println("Test 10")
end
export test_func

export solve!, simulate_moments

end
