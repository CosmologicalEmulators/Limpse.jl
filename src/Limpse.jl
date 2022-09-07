module Limpse

using Base: @kwdef
using SimpleChains

function maximin_input!(x, in_MinMax)
    for i in 1:length(x)
        x[i] -= in_MinMax[i,1]
        x[i] /= (in_MinMax[i,2]-in_MinMax[i,1])
    end
end

function inv_maximin_output!(x, out_MinMax)
    for i in 1:length(x)
        x[i] *= (out_MinMax[i,2]-out_MinMax[i,1])
        x[i] += out_MinMax[i,1]
    end
end

abstract type AbstractTrainedEmulators end

@kwdef mutable struct SimpleChainsEmulator <: AbstractTrainedEmulators
    Architecture
    Weights
end

abstract type AbstractPkEmulators end

@kwdef mutable struct PkEmulators <: AbstractPkEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::Array
    InMinMax::Matrix{Float64} = zeros(5,2)
    OutMinMax::Array{Float64} = zeros(2499,2)
end

function compute_Pk(input_params, PkEmulator::AbstractPkEmulators)
    H0 = input_params[3]*100
    ΩM = (input_params[4]+input_params[5])/(input_params[3]^2)
    As = exp(input_params[1])*1e-10
    ns = input_params[2]
    input = deepcopy(input_params[3:7])
    maximin_input!(input, PkEmulator.InMinMax)
    output = Array(run_emulator(input, PkEmulator.TrainedEmulator))
    inv_maximin_output!(output, PkEmulator.OutMinMax)
    return @. (P_prim(PkEmulator.kgrid, As, ns) * output ^2 * tilde_Δ(PkEmulator.kgrid, ΩM, H0)^2)
end

function compute_Tk(input_params, PkEmulator::AbstractPkEmulators)
    input = deepcopy(input_params[3:7])
    maximin_input!(input, PkEmulator.InMinMax)
    output = Array(run_emulator(input, PkEmulator.TrainedEmulator))
    inv_maximin_output!(output, PkEmulator.OutMinMax)
    return output
end


function run_emulator(input, trained_emulator::SimpleChainsEmulator)
    return trained_emulator.Architecture(input, trained_emulator.Weights)
end

function tilde_Δ(k, ΩM, H0)
    c0 = 2.99792458e5
    return @. (c0^2)*k^2/(1.5*ΩM*H0^2)
end

function P_prim(k, As, ns)
    return @. As * (k/0.05)^(ns-1)*2*π^2/(k^3)
end

function logspace(min, max, n)
    logmin = log10(min)
    logmax = log10(max)
    logarray = Array(LinRange(logmin, logmax, n))
    return exp10.(logarray)
end

end # module
