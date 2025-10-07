################################################################################
#################### Multi-Purpose #############################################
################################################################################
@concrete struct CobbDouglas{N}
    tfp
    coefs
end
CobbDouglas(N::Int) = CobbDouglas{N}(1.0, fill(1/N, N))
CobbDouglas(coefs; tfp = 1.0) = CobbDouglas{length(coefs)}(tfp, coefs)

(f::CobbDouglas{N})(inputs::Vararg{Any, N}) where {N} = f.tfp * prod(inputs .^ f.coefs)
marginal(f::CobbDouglas{N}, inputs::Vararg{Any, N}; wrt_position::Int) where {N} = f.tfp * f.coefs[wrt_position] * f(inputs...) / inputs[wrt_position]

@kwdef @concrete struct CES{N}
    tfp
    coefs
    substitution
end

(f::CES{N})(inputs::Vararg{Any, N}) where {N} = f.tfp * (sum(f.coefs .* (inputs .^ f.substitution))^(1/f.substitution))

################################################################################
#################### Matching ##################################################
################################################################################
match_from_unemployment(m, θ) = m(1, θ)
match_from_vacancy(m, θ) = m(θ^-1, 1)

@kwdef @concrete struct DHRW # den Haan, Ramey, and Watson (2000)
    coef = 0.5
end
(m::DHRW)(u, v) = (u*v)/((u^m.coef + v^m.coef)^(1/m.coef))
match_from_unemployment(m::DHRW, θ) = θ / (1 + θ^m.ℓ)^(1/m.coef)
match_from_vacancy(m::DHRW, θ) = 1 / (1 + θ^m.coef)^(1/m.coef)


match_from_unemployment(f::CobbDouglas, θ) = f.tfp*θ^(1-f.coefs[1])
match_from_vacancy(f::CobbDouglas, θ) = f.tfp*θ^-f.coefs[1]

################################################################################
#################### Utility ###################################################
################################################################################
@kwdef @concrete struct CRRA
    γ
end

function (u::CRRA)(x)
    if isone(u.γ)
        return log(x)
    else
        return x^(1-u.γ)/(1-u.γ)
    end
end

marginal(u::CRRA, x) = x^(-u.γ)


################################################################################
#################### Production ################################################
################################################################################
