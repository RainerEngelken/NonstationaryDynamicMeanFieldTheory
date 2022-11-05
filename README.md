# NonstationaryDynamicMeanFieldTheory
Code for calculating the autocorrelation and Lyapunov exponent of recurrent rate network

The provided script nonstationaryDMFT.jl calculates autocorrelations and largest Lyapunov exponent using DMFT of Gaussian relu network with common sine input.

When using, please cite the following work: 

"Input correlations impede suppression of chaos and learning in balanced rate networks
Rainer Engelken, Alessandro Ingrosso, Ramin Khajeh, Sven Goedeke*, L. F. Abbott*
to appear in PLOS Computational Biology"

The code is in Julia version 1.8.2 (https://julialang.org/)

EXAMPLE USE:

```include("include("nonstationaryDMFT.jl")```

```g, f, I₁, I₀, N, J₀, steps, bw, tol, dt, subDir = 1.6, 1.0, 0.8*√5000, 1.0, 5000, 1.0, 2^13,2^9,1e-2,2^-4, "rsmft01"```

```getLambdamaxCommon(g, f, I₁, I₀, N, J₀, steps, bw, tol, dt, subDir)```

Please send bug reports, issues & questions to the following email:
```echo "ude.aibmuloc@5632er" | rev```

