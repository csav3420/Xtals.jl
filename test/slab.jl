using Test
using Xtals
using LinearAlgebra
using Graphs

# Set data paths for crystal files
set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

@testset "Slab Construction" begin

    @testset "Miller Surface Basis" begin
        # Standard directions
        M100 = surface_basis((1, 0, 0))
        @test det(M100) > 0
        @test dot(M100[:, 1], [1, 0, 0]) == 0
        @test dot(M100[:, 2], [1, 0, 0]) == 0

        M010 = surface_basis((0, 1, 0))
        @test det(M010) > 0
        @test dot(M010[:, 1], [0, 1, 0]) == 0

        M001 = surface_basis((0, 0, 1))
        @test det(M001) > 0
        @test dot(M001[:, 1], [0, 0, 1]) == 0

        # Mixed Miller directions
        for hkl in [(1, 1, 0), (1, 1, 1), (1, 2, 3), (2, 1, 0), (-1, 1, 0)]
            M = surface_basis(hkl)
            @test det(M) > 0
            @test dot(M[:, 1], collect(hkl)) == 0
            @test dot(M[:, 2], collect(hkl)) == 0
        end

        # Error on (0, 0, 0)
        @test_throws ArgumentError surface_basis((0, 0, 0))
    end

    @testset "Cartesian z-Alignment & Dual Vacuum" begin
        xtal = Crystal("SBMOF-1.cif")
        slabs = make_slab(xtal, (1, 0, 0); vacuum=12.0, layers=1)
        slab = slabs[1]

        # Cell vectors u1, u2 in x-y plane (z ≈ 0)
        f2c = slab.box.f_to_c
        @test isapprox(f2c[3, 1], 0.0, atol=1e-5)
        @test isapprox(f2c[3, 2], 0.0, atol=1e-5)

        # Total cell height includes 2 * vacuum (24.0 Å)
        @test f2c[3, 3] >= 24.0
    end

    @testset "Cell Angle Orthogonalization" begin
        xtal = Crystal("SBMOF-1.cif")

        # orthogonalize_z = true (default) -> alpha=90°, beta=90°
        slabs_z = make_slab(xtal, (1, 0, 0); vacuum=10.0, orthogonalize_z=true)
        box_z = slabs_z[1].box
        @test isapprox(rad2deg(box_z.α), 90.0, atol=1e-4)
        @test isapprox(rad2deg(box_z.β), 90.0, atol=1e-4)

        # orthogonalize = true -> alpha=90°, beta=90°, gamma=90°
        slabs_ortho = make_slab(xtal, (1, 0, 0); vacuum=10.0, orthogonalize=true)
        box_ortho = slabs_ortho[1].box
        @test isapprox(rad2deg(box_ortho.α), 90.0, atol=1e-4)
        @test isapprox(rad2deg(box_ortho.β), 90.0, atol=1e-4)
        @test isapprox(rad2deg(box_ortho.γ), 90.0, atol=1e-4)
    end

    @testset "Surface Dimensions & Layer Series" begin
        xtal = Crystal("SBMOF-1.cif")

        # Target min surface size
        slabs = make_slab(xtal, (1, 0, 0); min_size=(15.0, 15.0), vacuum=10.0)
        box = slabs[1].box
        @test box.a >= 15.0
        @test box.b >= 15.0

        # Multi-layer series outputs
        slabs_series = make_slab(xtal, (1, 0, 0); max_height=30.0, min_size=(10.0, 10.0))
        @test length(slabs_series) >= 2
        @test slabs_series[1] isa Crystal
        @test slabs_series[2] isa Crystal

        # Alias make_slabs
        slabs_alias = make_slabs(xtal, (1, 0, 0); max_height=30.0)
        @test length(slabs_alias) == length(slabs_series)
    end

    @testset "Molecular Centroids, Terminations & Bond Preservation" begin
        xtal = Crystal("SBMOF-1.cif")
        infer_bonds!(xtal, true)

        # Detect molecules
        mols = detect_molecules(xtal)
        @test !isempty(mols)

        # Terminations
        terms = get_terminations(xtal, (1, 0, 0))
        @test !isempty(terms)

        # Make slab and check preserved bond edges
        slabs = make_slab(xtal, (1, 0, 0); vacuum=10.0, termination=1, keep_molecules_whole=true)
        slab = slabs[1]
        @test ne(slab.bonds) > 0
    end

    @testset "Aspirin Slabs & Multi-Termination Verification" begin
        aspirin = Crystal("aspirin.cif")
        infer_bonds!(aspirin, true)
        @test aspirin.atoms.n == 84

        # Verify multi-termination slab generation across multiple Miller indices
        for hkl in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 3)]
            terms = get_terminations(aspirin, hkl)
            @test !isempty(terms)
            for (t_idx, term) in enumerate(terms)
                slabs = make_slab(aspirin, hkl; termination=t_idx, vacuum=10.0, layers=1)
                @test slabs[1].atoms.n == 84
                @test ne(slabs[1].bonds) == 84
                # Verify physical inter-atomic distances (no overlapping atoms)
                min_d = check_slab_distances(slabs[1])
                @test min_d >= 1.4
            end
        end

        # Termination equivalence check
        terms_100 = get_terminations(aspirin, (1, 0, 0))
        if length(terms_100) >= 2
            @test !are_terminations_equivalent(aspirin, (1, 0, 0), terms_100[1], terms_100[2])
            @test are_terminations_equivalent(aspirin, (1, 0, 0), terms_100[1], terms_100[1])
        end

        # Test unique_only parameter in get_terminations (Strict 2-termination assertions for Aspirin 001 and 010)
        u_terms_001 = get_terminations(aspirin, (0, 0, 1); unique_only=true)
        @test length(u_terms_001) == 2

        u_terms_010 = get_terminations(aspirin, (0, 1, 0); unique_only=true)
        @test length(u_terms_010) == 2

        # Test High Miller Index Minimal Basis (det(M)=1) & Gap-Free Supercell Size Control
        for hkl_high in [(2, 1, 2), (3, 1, 2), (3, 2, 1)]
            M_high = Xtals.surface_basis(hkl_high; f_to_c=aspirin.box.f_to_c)
            @test abs(round(Int, det(M_high))) == 1

            slabs_high = make_slab(aspirin, hkl_high; min_size=(15.0, 15.0), vacuum=10.0, layers=1)
            @test length(slabs_high) >= 1
            min_d_high = check_slab_distances(slabs_high[1])
            @test min_d_high >= 1.4
        end
    end

    @testset "Surface Dipole & Tasker Surface Classification" begin
        # 1. Point charge dipole math test
        box_test = Box([10.0 0.0 0.0; 0.0 10.0 0.0; 0.0 0.0 10.0])
        coords_cart = Cart([0.0 0.0; 0.0 0.0; 2.0 0.0]) # +1.0 at z=2.0 Å, -1.0 at z=0.0 Å
        coords_frac = Frac(coords_cart, box_test)
        atoms_test = Atoms([:Na, :Cl], coords_frac)
        charges_test = Charges([1.0, -1.0], coords_frac)
        dipole_xtal = Crystal("dipole_test", box_test, atoms_test, charges_test)

        mu = dipole_moment(dipole_xtal)
        @test isapprox(mu, [0.0, 0.0, 2.0], atol=1e-5)

        mu_debye = dipole_moment(dipole_xtal; debye=true)
        @test isapprox(mu_debye, [0.0, 0.0, 2.0 * 4.803204], atol=1e-4)

        @test surface_dipole_moment(dipole_xtal) == 2.0
        @test is_polar_surface(dipole_xtal) == true

        # 2. Aspirin non-polar surface slab tests
        aspirin_xtal = Crystal("aspirin.cif")
        aspirin_chg = assign_charges(aspirin_xtal, Dict(:C => 0.1, :H => 0.05, :O => -0.325), Inf)
        aspirin_slabs = make_slab(aspirin_chg, (0, 0, 1); vacuum=10.0, layers=1)
        @test isapprox(surface_dipole_moment(aspirin_slabs[1]), 0.0, atol=1e-4)
        @test is_polar_surface(aspirin_slabs[1]) == false
        @test tasker_type(aspirin_chg, (0, 0, 1)) == :Type_I

        # 3. get_terminations nonpolar_only option
        np_terms = get_terminations(aspirin_chg, (0, 0, 1); unique_only=true, nonpolar_only=true)
        @test length(np_terms) == 2
    end

    @testset "Error Handling & Input Validation" begin
        xtal = Crystal("SBMOF-1.cif")
        box_test = Box([10.0 0.0 0.0; 0.0 10.0 0.0; 0.0 0.0 10.0])
        coords_frac = Frac(Cart([0.0 0.0; 0.0 0.0; 0.0 0.0]), box_test)
        atoms_test = Atoms([:C, :H], coords_frac)
        no_chg_xtal = Crystal("no_chg", box_test, atoms_test, Charges([0.0, 0.0], coords_frac))

        @test_throws ArgumentError make_slab(xtal, (0, 0, 0))
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); vacuum=-5.0)
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); max_height=-10.0)
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); min_thickness=-1.0)
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); min_size=(-5.0, 10.0))
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); layers=0)
        @test_throws ArgumentError dipole_moment(no_chg_xtal) # Zero/unassigned charges exception
    end

end
