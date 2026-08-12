using Test, Xtals, Aqua, JuliaFormatter

@testset "Aqua.jl & Code Formatting Quality Assurance" begin
    Aqua.test_all(Xtals; ambiguities=false, deps_compat=false, stale_deps=(ignore=[:JLD2],))
    @test JuliaFormatter.format(Xtals; verbose=false)
end
