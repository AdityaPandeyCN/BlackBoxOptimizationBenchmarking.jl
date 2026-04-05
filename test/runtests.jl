using Test
using BlackBoxOptimizationBenchmarking, Plots, Optimization
import BlackBoxOptimizationBenchmarking: Chain, BenchmarkSetup, bbob_suite
const BBOB = BlackBoxOptimizationBenchmarking

using OptimizationBBO, OptimizationOptimJL

##

const D = 3
test_functions = bbob_suite(Val(D))

@testset "Function optima" begin
    for f in test_functions
        @test f(f.x_opt) ≈ f.f_opt atol=1e-5
    end
end

##

b = BBOB.benchmark(
    NelderMead(), test_functions[1], [100, 500, 1000], 
)

b = BBOB.benchmark(
    BenchmarkSetup(NelderMead(); isboxed=false), test_functions[1], [100, 500, 1000], 
)

@test length(b.success_count) == 3

b1 = BBOB.benchmark(
    NelderMead(), test_functions, [100, 500, 1000],
)

b2 = BBOB.benchmark(
    ParticleSwarm(), test_functions, [100, 500, 1000],
)

plot(b1; label = "NelderMead")
plot!(b2; label = "ParticleSwarm")

## OptimizationBBO

setup = Chain(
    BenchmarkSetup(BBO_adaptive_de_rand_1_bin(), isboxed = true),
    BenchmarkSetup(NelderMead(), isboxed = false),
    0.9,
)

b = BBOB.benchmark(
    setup, test_functions, [1000, 5000, 10000],
)

plot(b)

##

plot_functions = bbob_suite(Val(2))
plot(plot_functions[1])

##