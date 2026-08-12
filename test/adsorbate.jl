using Test
using Xtals
using LinearAlgebra

set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

@testset "Surface Adsorbate & Solvation Overlayer Tests" begin
    aspirin_xtal = Crystal("aspirin.cif")
    aspirin_slab = make_slab(aspirin_xtal, (1, 1, 0); vacuum=10.0, layers=1)[1]
    n_slab_atoms = aspirin_slab.atoms.n
    n_mol_atoms = length(detect_molecules(aspirin_xtal)[1].species) # 1 Aspirin molecule = 21 atoms

    # =========================================================================================
    # 1. STRUCTURAL INTEGRITY & ZERO ATOM OVERLAP
    # =========================================================================================
    @testset "Structural Integrity & Overlap Prevention" begin
        ads_mol_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; coverage=(1, 1), n_trials=5
        )
        @test ads_mol_slab.atoms.n == n_slab_atoms + n_mol_atoms

        # Zero atom overlaps assertion
        overlap_flag, overlap_pairs = overlap(ads_mol_slab, false)
        @test !overlap_flag
        @test isempty(overlap_pairs)

        # File path string adsorbate deposition (.cif)
        ads_file_xtal = read_adsorbate(joinpath(rc[:paths][:crystals], "aspirin.cif"))
        n_file_mol_atoms = length(detect_molecules(ads_file_xtal)[1].species)
        ads_file_slab = make_adsorbate_slab(
            aspirin_slab,
            joinpath(rc[:paths][:crystals], "aspirin.cif");
            coverage=(1, 1),
            n_trials=5,
        )
        @test ads_file_slab.atoms.n == n_slab_atoms + n_file_mol_atoms
        overlap_flag_file, _ = overlap(ads_file_slab, false)
        @test !overlap_flag_file
    end

    # =========================================================================================
    # 2. ATOM COUNT SCALING & FILM THICKNESS (layer_height)
    # =========================================================================================
    @testset "Atom Count Scaling & Film Height" begin
        # 2x2 coverage grid = 4 adsorbates
        ads_grid_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; coverage=(2, 2), n_trials=5
        )
        @test ads_grid_slab.atoms.n == n_slab_atoms + (4 * n_mol_atoms)

        # 2 discrete layers
        ads_2layer_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; coverage=(1, 1), n_layers=2, n_trials=5
        )
        @test ads_2layer_slab.atoms.n == n_slab_atoms + (2 * n_mol_atoms)

        # Film height target (layer_height = 10.0 Å)
        ads_10a_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; coverage=(1, 1), layer_height=10.0, n_trials=5
        )
        @test ads_10a_slab.atoms.n > n_slab_atoms + n_mol_atoms
        @test ads_10a_slab.box.f_to_c[3, 3] > aspirin_slab.box.f_to_c[3, 3]
    end

    # =========================================================================================
    # 3. SELECTION MODES & IN-PLACE ALIAS
    # =========================================================================================
    @testset "Candidate Selection Modes & Aliases" begin
        ads_energy_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; selector=:energy, n_trials=5
        )
        @test ads_energy_slab.atoms.n == n_slab_atoms + n_mol_atoms

        ads_contact_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; selector=:contact, n_trials=5
        )
        @test ads_contact_slab.atoms.n == n_slab_atoms + n_mol_atoms

        # Alias add_adsorbate!
        ads_alias_slab = add_adsorbate!(
            aspirin_slab, aspirin_xtal; coverage=(1, 1), n_trials=5
        )
        @test ads_alias_slab.atoms.n == n_slab_atoms + n_mol_atoms
    end

    # =========================================================================================
    # 4. PACKAGE SANITY CHECKS & INPUT GUARDS
    # =========================================================================================
    @testset "Package Sanity Checks & Input Guards" begin
        @test_throws ArgumentError make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; vacuum=-5.0
        )
        @test_throws ArgumentError make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_trials=0
        )
        @test_throws ArgumentError make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; layer_height=-10.0
        )
        @test_throws ArgumentError make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_layers=0
        )

        # Coordinate dimension sanity
        ads_test = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; coverage=(1, 1), n_trials=5
        )
        @test size(ads_test.atoms.coords.xf, 1) == 3
        @test size(ads_test.atoms.coords.xf, 2) == ads_test.atoms.n
        @test length(ads_test.atoms.species) == ads_test.atoms.n
        @test length(ads_test.charges.q) == ads_test.atoms.n
    end

    # =========================================================================================
    # 5. PHYSICAL & THERMODYNAMIC VALIDATION TESTS
    # =========================================================================================
    @testset "Physical & Thermodynamic Validation" begin
        ads_phys_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; vacuum=10.0, n_trials=5
        )
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
