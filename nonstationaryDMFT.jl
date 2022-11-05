# This script calculates autocorrelations and largest Lyapunov exponent using DMFT of Gaussian relu network with common sine input.

# When using, please cite the following work: 
#
#   "Input correlations impede suppression of chaos and learning in balanced rate networks
#   Rainer Engelken, Alessandro Ingrosso, Ramin Khajeh, Sven Goedeke*, L. F. Abbott*
#   to appear in PLOS Computational Biology"
#
# The code is in Julia version 1.8.2 (https://julialang.org/)
#
# COPYRIGHT: Rainer Engelken 2022
# please send bugs reports, questions and issues to re236<at>columbia<dot>edu 
# This code is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# <http://www.gnu.org/licenses/>


#= EXAMPLE USE
g, f, I₁, I₀, N, J₀, steps, bw, tol, dt, subDir = 1.6, 1.0, 0.8*√5000, 1.0, 5000, 1.0, 2^13,2^9,1e-2,2^-4, "rsmft01"
getLambdamaxCommon(g, f, I₁, I₀, N, J₀, steps, bw, tol, dt, subDir)
=#

using Roots,SpecialFunctions,DifferentialEquations,QuadGK,Images
include("gaussintegrals.jl")
######################################## DDMFT for common input #########################################

relu(x) = max(0, x)
function getLambdamaxCommon(g, f, I₁, I₀, N, J₀, steps, bw, tol, dt, subDir)
    println(" \n calculating AC and Lyapunov exponent for I₁:$I₁" * " and g:$g  f:$f \n")
    tStart = time_ns()
    timeAfterStart() = round((time_ns() - tStart) / 1e9, digits = 1)
    nDisplay = 1000
    cstart, mstart = mc₀analytical(g, J₀, I₀, N)
    function m_ode(m, c, t)
        ν = √c * relugauss(m / √c)
        return -m - √N * J₀ * ν + I₀ * √N + I₁ * sinpi(2f * t) # Eq. 2a
    end
    tspan = (0.0, dt * steps)
    prob = ODEProblem(m_ode, mstart, tspan, cstart)
    integrator = init(prob, Rosenbrock23(); adaptive = false, dt = dt, reltol = tol, abstol = tol)
    rateAll, diagr, cend = Float64[], Float64[], Float64[]
    for x in [:m, :c, :r, :k, :l, :qϕ′, :qϕ, :rOld, :lOld]
        @eval $x = zeros(steps)
    end
    c[1] = cstart
    m[1] = mstart
    diagc = [cstart]
    diagk = [0.0]
    nAll = 0
    nsteptotal = round(Int, 1 / 2 * (bw + 1) * (2steps - bw))
    tStartSim = time_ns()
    timeAfterSimStart() = round((time_ns() - tStart) / 1e9, digits = 1)

    @time for n2 = 1:steps-1
        if n2 == div(steps, 2)
            k[n2] = 1.0
        end
        n2 % nDisplay == 0 && print("\r n2:", n2, " ")
        n2 % nDisplay == 0 && println("\r", round(nAll * 100 / nsteptotal, digits = 1), " % after ", timeAfterSimStart(), " s. Left:", round(Int, (nsteptotal - nAll + 1) * timeAfterSimStart() / (nAll - 1)), " s. SimTime: ", round(Int, dt * n2), " τ. n2: ", n2)
        rate = √diagc[n2] * relugauss(m[n2] / √diagc[n2])
        push!(rateAll, rate)

        integrator.p = diagc[n2]
        step!(integrator)
        m[n2+1] = integrator.sol.u[end]

        diagcNp1 = (1 - dt)^2 * diagc[n2] + 2 * (1 - dt) * dt * r[n2] + dt^2 * g^2 * diagc[n2] * relugauss2(m[n2] / √diagc[n2])
        push!(diagc, diagcNp1)
        if n2 >= div(steps, 2)
            k[n2+1] = (1 - dt)^2 * k[n2] + 2 * (1 - dt) * dt * l[n2] + dt^2 * g^2 * stepgaussintegral(m[n2] / √diagc[n2]) * k[n2]

            push!(diagk, k[n2+1])
        else
            push!(diagk, 0.0)
        end

        for n1 = max(1, n2 - bw):n2
            nAll += 1
            if n1 == max(1, n2 - bw)
                r[n1] = 0.0
                l[n1] = 0.0
            end
            c[n1] = (1 - dt) * c[n1] + dt * rOld[n1] # Eq. 10
        end
        Threads.@threads for n1 = max(1, n2 - bw):n2
            qϕ[n1] = get_qϕ(c[n1], diagc[n1], diagc[n2+1], relu, m[n1], m[n2+1], tol, tol) # Eq. 9
        end

        for n1 = max(1, n2 - bw):n2
            r[n1+1] = (1 - dt) * r[n1] + dt * g^2 * qϕ[n1] # Eq. 11
        end
        for n1 = max(1, n2 - bw):n2
            if n2 >= div(steps, 2)
                k[n1] = (1 - dt) * k[n1] + dt * lOld[n1] # Eq. 14
            end
        end
        Threads.@threads for n1 = max(1, n2 - bw):n2
            if n2 >= div(steps, 2)
                qϕ′[n1+1] = get_qϕ′(c[n1], diagc[n1], diagc[n2+1], m[n1], m[n2+1], tol, tol) * k[n1] # Eq. 13
            end
        end
        for n1 = max(1, n2 - bw):n2
            if n2 >= div(steps, 2)
                l[n1+1] = (1 - dt) * l[n1] + dt * g^2 * qϕ′[n1+1] # Eq. 15
            end
        end
        c[n2+1] = diagc[n2+1]
        k[n2+1] = diagk[n2+1]
        rOld .= r
        lOld .= l
    end

    linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y
    u = log.(abs.(diagk)) / 2 # Eq. 16
    idmaxAll = map(i -> i[1], findlocalmaxima(u, 1, false))

    idmax = Int64[]
    for id in idmaxAll
        if 3 < id < length(u) - 1
            u[id-2] <= u[id-1] && u[id+1] >= u[id+2] && push!(idmax, id)
        end
    end
    ap, bp = linreg(idmax[idmax.>div(3length(u), 4)] .- div(3length(u), 4), u[idmax[idmax.>div(3length(u), 4)]])
    a, b = linreg(1:length(u[div(3 * end, 4)+1:end]), u[div(3 * end, 4)+1:end])
    if length(idmax[idmax.>div(3length(u), 4)]) > 1
        println("Lyapunov exponent defined by average slope of peaks of log of diagonal of k")
        a, b = ap, bp
    end
    lle = b / dt
    return lle
end
