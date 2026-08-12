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
        xtal = Crystal("aspirin.cif")
        slabs = make_slab(xtal, (1, 0, 0); vacuum=100.0, layers=1)
        slab = slabs[1]

        # Cell vectors u1, u2 in x-y plane (z ≈ 0)
        f2c = slab.box.f_to_c
        @test isapprox(f2c[3, 1], 0.0, atol=1e-5)
        @test isapprox(f2c[3, 2], 0.0, atol=1e-5)

        # Total cell height includes 2 * vacuum (200.0 Å)
        @test f2c[3, 3] >= 200.0
    end

    @testset "Cell Angle Orthogonalization" begin
        xtal = Crystal("aspirin.cif")

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
        xtal = Crystal("aspirin.cif")

        # Target min surface size
        slabs = make_slab(xtal, (1, 0, 0); min_size=(25.0, 50.0), vacuum=10.0)
        box = slabs[1].box
        @test box.a >= 25.0
        @test box.b >= 50.0

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
        xtal = Crystal("aspirin.cif")
        infer_bonds!(xtal, true)

        # Detect molecules
        mols = detect_molecules(xtal)
        @test !isempty(mols)

        # Terminations
        terms = get_terminations(xtal, (1, 0, 0))
        @test !isempty(terms)

        # Make slab and check preserved bond edges
        slabs = make_slab(
            xtal, (1, 0, 0); vacuum=10.0, termination=1, keep_molecules_whole=true
        )
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
            @test !are_terminations_equivalent(
                aspirin, (1, 0, 0), terms_100[1], terms_100[2]
            )
            @test are_terminations_equivalent(
                aspirin, (1, 0, 0), terms_100[1], terms_100[1]
            )
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

            slabs_high = make_slab(
                aspirin, hkl_high; min_size=(15.0, 15.0), vacuum=10.0, layers=1
            )
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

        # 2. Aspirin non-polar surface slab tests & layer dipoles
        aspirin_xtal = Crystal("aspirin.cif")
        aspirin_chg = assign_charges(
            aspirin_xtal, Dict(:C => 0.1, :H => 0.05, :O => -0.325), Inf
        )
        aspirin_slabs = make_slab(aspirin_chg, (0, 0, 1); vacuum=10.0, layers=1)
        @test isapprox(surface_dipole_moment(aspirin_slabs[1]), 0.0, atol=1e-4)
        @test is_polar_surface(aspirin_slabs[1]) == false
        @test tasker_type(aspirin_chg, (0, 0, 1)) == :Type_I

        # 3. Tasker Type II classification & layer_dipoles for Aspirin (1, 0, 0)
        ld_100 = layer_dipoles(aspirin_chg, (1, 0, 0))
        @test length(ld_100) == 4
        @test abs(ld_100[1].mu_z) > 0.5 # Non-zero sub-layer dipole!
        @test tasker_type(aspirin_chg, (1, 0, 0)) == :Type_II

        # 4. get_terminations nonpolar_only option
        np_terms = get_terminations(
            aspirin_chg, (0, 0, 1); unique_only=true, nonpolar_only=true
        )
        @test length(np_terms) == 2

        # 5. Independent dual top and bottom surface termination test
        dual_slabs_tuple = make_slab(
            aspirin_xtal, (1, 1, 0); vacuum=10.0, layers=1, termination=(1, 2)
        )
        @test length(dual_slabs_tuple) >= 1

        dual_slabs_kwargs = make_slab(
            aspirin_xtal,
            (1, 1, 0);
            vacuum=10.0,
            layers=1,
            termination_bottom=1,
            termination_top=2,
        )
        @test length(dual_slabs_kwargs) >= 1
        @test dual_slabs_tuple[1].atoms.n == dual_slabs_kwargs[1].atoms.n
    end

    @testset "Geometric Reference Assertions (Aspirin (1,1,0) Slab)" begin
        aspirin_xtal = Crystal("aspirin.cif")
        ref_slab = make_slab(aspirin_xtal, (1, 1, 0); vacuum=10.0, layers=1)[1]

        # Test 1: Molecule Count Reference
        ref_mols = detect_molecules(ref_slab)
        @test length(ref_mols) == 4

        # Test 2: Total Atom Count & Species Composition Reference
        @test ref_slab.atoms.n == 84
        @test count(==(Symbol("C")), ref_slab.atoms.species) == 36
        @test count(==(Symbol("H")), ref_slab.atoms.species) == 32
        @test count(==(Symbol("O")), ref_slab.atoms.species) == 16

        # Test 3: Molecular Centroids Reference
        ref_centroids = sort(
            [round.(m.centroid_cart, digits=4) for m in ref_mols], by=x->x[3]
        )
        expected_centroids = [
            [-4.6281, -9.3872, 10.0501],
            [-11.4318, -0.4829, 13.7004],
            [-10.3761, -7.2278, 13.8006],
            [-5.6838, 0.6297, 15.6328],
        ]
        expected_centroids_sorted = sort(expected_centroids, by=x->x[3])
        for (c_calc, c_exp) in zip(ref_centroids, expected_centroids_sorted)
            @test isapprox(c_calc, c_exp, atol=1e-3)
        end

        # Test 4: Cell Vector Lengths Reference (a, b, c)
        a_len = norm(ref_slab.box.a)
        b_len = norm(ref_slab.box.b)
        c_len = norm(ref_slab.box.c)
        @test isapprox(a_len, 11.496000, atol=1e-4)
        @test isapprox(b_len, 13.912005, atol=1e-4)
        @test isapprox(c_len, 25.682860, atol=1e-4)

        # Test 5: Cell Angles Reference (alpha, beta, gamma)
        alpha_deg = ref_slab.box.α * (180.0 / pi)
        beta_deg = ref_slab.box.β * (180.0 / pi)
        gamma_deg = ref_slab.box.γ * (180.0 / pi)
        @test isapprox(alpha_deg, 90.0, atol=1e-4)
        @test isapprox(beta_deg, 90.0, atol=1e-4)
        @test isapprox(gamma_deg, 70.849289, atol=1e-4)

        # Test 6: Slab Height Span & Vacuum Padding Reference
        c_cart = Cart(ref_slab.atoms.coords, ref_slab.box).x
        min_z = minimum(c_cart[3, :])
        max_z = maximum(c_cart[3, :])
        @test isapprox(min_z, 6.7908, atol=1e-3)
        @test isapprox(max_z, 18.8464, atol=1e-3)
        @test isapprox(max_z - min_z, 12.0556, atol=1e-3)
    end

    @testset "Error Handling & Input Validation" begin
        xtal = Crystal("aspirin.cif")
        box_test = Box([10.0 0.0 0.0; 0.0 10.0 0.0; 0.0 0.0 10.0])
        coords_frac = Frac(Cart([0.0 0.0; 0.0 0.0; 0.0 0.0]), box_test)
        atoms_test = Atoms([:C, :H], coords_frac)
        no_chg_xtal = Crystal(
            "no_chg", box_test, atoms_test, Charges([0.0, 0.0], coords_frac)
        )

        @test_throws ArgumentError make_slab(xtal, (0, 0, 0))
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); vacuum=-5.0)
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); max_height=-10.0)
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); min_thickness=-1.0)
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); min_size=(-5.0, 10.0))
        @test_throws ArgumentError make_slab(xtal, (1, 0, 0); layers=0)
        @test_throws ArgumentError dipole_moment(no_chg_xtal) # Zero/unassigned charges exception
    end
end
