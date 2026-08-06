using Test, Xtals, Aqua

@testset "Aqua.jl Quality Assurance" begin
    Aqua.test_all(
        Xtals;
        ambiguities=false,
        deps_compat=false,
        stale_deps=(ignore=[:JLD2],)
    )
end
