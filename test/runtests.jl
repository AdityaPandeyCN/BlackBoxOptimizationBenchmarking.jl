using Test
using BlackBoxOptimizationBenchmarking, Plots, Optimization
import BlackBoxOptimizationBenchmarking: Chain, BenchmarkSetup, bbob_suite
const BBOB = BlackBoxOptimizationBenchmarking

using OptimizationBBO, OptimizationOptimJL

##

# Create test suite for dimension 3
const D = 3
test_functions = bbob_suite(Val(D))

# Test that functions evaluate correctly at optimum
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

b = BBOB.benchmark(
    NelderMead(), test_functions[1:3], 100:100:2000, Ntrials=10,
)

b2 = BBOB.benchmark(
    ParticleSwarm(), test_functions[1:3], 100:200:2000, Ntrials=10,
)

BBOB.compute_CI!(b, 0.1)
plot(b; title = "Benchmark", label = "NelderMead", legend = :outerright)
plot!(b2; label = "ParticleSwarm")

## OptimizationBBO

setup = Chain(
    BenchmarkSetup(BBO_adaptive_de_rand_1_bin(), isboxed = true),
    BenchmarkSetup(NelderMead(), isboxed = false),
    0.9
)

b = BBOB.benchmark(
    setup, test_functions[1:2], 100:200:5000, Ntrials=10,
)

plot(b)
