
stepgaussintegral(x) = (1 + erf(x / √2)) / 2
relugauss(x) = (x + √(2 / π) * exp(-x^2 / 2) + x * erf(x / √2)) / 2
relugauss2(x) = x * (exp(-x^2 / 2) / √(2π)) + ((1 + x^2) * (1 + erf(x / √2))) / 2
q0(J₀, I₀, N, x) = (I₀ * √N / (x + √N * J₀ * relugauss(x)))^2

function mc₀analytical(g, J₀, I₀, N)
    x = fzero(x -> g^2 * relugauss2(x) - 1, 0.0)::Float64
    q₀ = q0(J₀, I₀, N, x)
    m = x * √q₀
    return q₀, m
end

function qϕinnerintegrand(x1, ρ, z)
    (x1 + ρ * z) * (1 + erf((x1 + ρ * z) / (√2 * √relu(1 - ρ^2)))) / 2 + √relu(1 - ρ^2) * exp(-(x1 + ρ * z)^2 / (2 * relu(1 - ρ^2))) / √(2π)
end

function qϕintegrand(c, c₀₁, c₀₂, m₁, m₂, z)
    ρ = c / √(c₀₁ * c₀₂)
    (m₂ / √c₀₂ + z) * qϕinnerintegrand(m₁ / √c₀₁, ρ, z) * exp(-z^2 / 2) / √(2π)
end

function get_qϕ(c, c₀₁, c₀₂, ϕ, m₁, m₂, reltolgauss, atolgauss)
    integral, ~ = quadgk(z -> qϕintegrand(c, c₀₁, c₀₂, m₁, m₂, z), -m₂ / √c₀₂, Inf, rtol = reltolgauss, atol = atolgauss)
    return integral * √(c₀₁ * c₀₂)
end

function qϕ′innerintegrand(x1, ρ, z)
    (1 + erf((x1 + ρ * z) / (√2 * √relu(1 - ρ^2)))) / 2
end

function qϕ′integrand(c, c₀₁, c₀₂, m₁, m₂, z)
    ρ = c / √(c₀₁ * c₀₂)
    qϕ′innerintegrand(m₁ / √c₀₁, ρ, z) * exp(-z^2 / 2) / √(2π)
end

function get_qϕ′(c, c₀₁, c₀₂, m₁, m₂, reltolgauss, atolgauss)
    integral, err1 = quadgk(z -> qϕ′integrand(c, c₀₁, c₀₂, m₁, m₂, z), -m₂ / √c₀₂, Inf, rtol = reltolgauss, atol = atolgauss)
    return integral
end
