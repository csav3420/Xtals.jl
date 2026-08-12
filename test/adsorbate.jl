using Test
using Xtals
using LinearAlgebra

set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

@testset "Surface Adsorbate & Solvation Overlayer Tests" begin
    aspirin_xtal = Crystal("aspirin.cif")
    aspirin_slab = make_slab(aspirin_xtal, (1, 1, 0); vacuum=10.0, layers=1)[1]
    n_slab_atoms = aspirin_slab.atoms.n

    # =========================================================================================
    # 1. STRUCTURAL INTEGRITY & ZERO ATOM OVERLAP
    # =========================================================================================
    @testset "Structural Integrity & Overlap Prevention" begin
        ads_o_slab = make_adsorbate_slab(aspirin_slab, :O; coverage=(1, 1), n_trials=10)
        @test ads_o_slab.atoms.n == n_slab_atoms + 1
        @test ads_o_slab.atoms.species[end] == :O

        # Zero atom overlaps assertion
        overlap_flag, overlap_pairs = overlap(ads_o_slab, false)
        @test !overlap_flag
        @test isempty(overlap_pairs)

        # Molecular adsorbate deposition (1 aspirin molecule = 21 atoms)
        ads_mol_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; coverage=(1, 1), n_trials=5
        )
        @test ads_mol_slab.atoms.n == n_slab_atoms + 21
        overlap_flag_mol, _ = overlap(ads_mol_slab, false)
        @test !overlap_flag_mol
    end

    # =========================================================================================
    # 2. ATOM COUNT SCALING & FILM THICKNESS (layer_height)
    # =========================================================================================
    @testset "Atom Count Scaling & Film Height" begin
        # 2x2 coverage grid = 4 adsorbates
        ads_grid_slab = make_adsorbate_slab(aspirin_slab, :O; coverage=(2, 2), n_trials=10)
        @test ads_grid_slab.atoms.n == n_slab_atoms + 4

        # 2 discrete layers
        ads_2layer_slab = make_adsorbate_slab(
            aspirin_slab, :H; coverage=(1, 1), n_layers=2, n_trials=10
        )
        @test ads_2layer_slab.atoms.n == n_slab_atoms + 2

        # Film height target (layer_height = 10.0 Å)
        ads_10a_slab = make_adsorbate_slab(
            aspirin_slab, :O; coverage=(1, 1), layer_height=10.0, n_trials=10
        )
        @test ads_10a_slab.atoms.n > n_slab_atoms + 1
        @test ads_10a_slab.box.f_to_c[3, 3] > aspirin_slab.box.f_to_c[3, 3]
    end

    # =========================================================================================
    # 3. SELECTION MODES & IN-PLACE ALIAS
    # =========================================================================================
    @testset "Candidate Selection Modes & Aliases" begin
        ads_energy_slab = make_adsorbate_slab(
            aspirin_slab, :O; selector=:energy, n_trials=10
        )
        @test ads_energy_slab.atoms.n == n_slab_atoms + 1

        ads_contact_slab = make_adsorbate_slab(
            aspirin_slab, :O; selector=:contact, n_trials=10
        )
        @test ads_contact_slab.atoms.n == n_slab_atoms + 1

        # Alias add_adsorbate!
        ads_alias_slab = add_adsorbate!(aspirin_slab, :O; coverage=(1, 1), n_trials=5)
        @test ads_alias_slab.atoms.n == n_slab_atoms + 1
    end

    # =========================================================================================
    # 4. PACKAGE SANITY CHECKS & INPUT GUARDS
    # =========================================================================================
    @testset "Package Sanity Checks & Input Guards" begin
        @test_throws ArgumentError make_adsorbate_slab(aspirin_slab, :O; vacuum=-5.0)
        @test_throws ArgumentError make_adsorbate_slab(aspirin_slab, :O; n_trials=0)
        @test_throws ArgumentError make_adsorbate_slab(aspirin_slab, :O; layer_height=-10.0)
        @test_throws ArgumentError make_adsorbate_slab(aspirin_slab, :O; n_layers=0)

        # Coordinate dimension sanity
        ads_test = make_adsorbate_slab(aspirin_slab, :O; coverage=(1, 1), n_trials=5)
        @test size(ads_test.atoms.coords.xf, 1) == 3
        @test size(ads_test.atoms.coords.xf, 2) == ads_test.atoms.n
        @test length(ads_test.atoms.species) == ads_test.atoms.n
        @test length(ads_test.charges.q) == ads_test.atoms.n
    end

    # =========================================================================================
    # 5. PHYSICAL & THERMODYNAMIC VALIDATION TESTS
    # =========================================================================================
    @testset "Physical & Thermodynamic Validation" begin
        ads_phys_slab = make_adsorbate_slab(aspirin_slab, :O; vacuum=10.0, n_trials=10)
        c_cart = Cart(ads_phys_slab.atoms.coords, ads_phys_slab.box).x
        z_coords = c_cart[3, :]

        # Vacuum separation check
        z_min = minimum(z_coords)
        z_max = maximum(z_coords)
        z_box = ads_phys_slab.box.f_to_c[3, 3]

        @test z_min >= 10.0 - 1e-4
        @test (z_box - z_max) >= 10.0 - 1e-4

        # Total charge conservation
        @test sum(ads_phys_slab.charges.q) ≈ sum(aspirin_slab.charges.q) atol=1e-5
    end
end
