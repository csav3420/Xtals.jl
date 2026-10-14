using Test
using Xtals
using Xtals: SVector
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
        # Exact molecule count deposition (n_molecules=1)
        ads_1mol_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_molecules=1, n_trials=10
        )
        @test ads_1mol_slab.atoms.n == n_slab_atoms + n_mol_atoms

        # Zero atom overlaps assertion
        overlap_flag, overlap_pairs = overlap(ads_1mol_slab, false)
        @test !overlap_flag
        @test isempty(overlap_pairs)

        # File path string adsorbate deposition (.cif)
        ads_file_xtal = read_adsorbate(joinpath(rc[:paths][:crystals], "aspirin.cif"))
        n_file_mol_atoms = length(detect_molecules(ads_file_xtal)[1].species)
        ads_file_slab = make_adsorbate_slab(
            aspirin_slab,
            joinpath(rc[:paths][:crystals], "aspirin.cif");
            n_molecules=1,
            n_trials=10,
        )
        @test ads_file_slab.atoms.n == n_slab_atoms + n_file_mol_atoms
        overlap_flag_file, _ = overlap(ads_file_slab, false)
        @test !overlap_flag_file
    end

    # =========================================================================================
    # 2. MOLECULE COUNT SCALING & FILM THICKNESS (layer_height)
    # =========================================================================================
    @testset "Molecule Count Scaling & Film Height" begin
        # Exact 3 molecules deposition
        ads_3mol_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_molecules=3, n_trials=10
        )
        @test ads_3mol_slab.atoms.n == n_slab_atoms + (3 * n_mol_atoms)

        # Zero atom overlap assertion for multi-molecule film
        overlap_flag_3, _ = overlap(ads_3mol_slab, false)
        @test !overlap_flag_3

        # Film height target (layer_height = 10.0 Å)
        ads_10a_slab = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; layer_height=10.0, n_trials=10
        )
        @test ads_10a_slab.atoms.n > n_slab_atoms + n_mol_atoms
        @test ads_10a_slab.box.f_to_c[3, 3] > aspirin_slab.box.f_to_c[3, 3]

        # Reproducible RNG seed test (seed=42)
        ads_seed1 = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_molecules=2, n_trials=10, seed=42
        )
        ads_seed2 = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_molecules=2, n_trials=10, seed=42
        )
        @test ads_seed1.atoms.coords.xf ≈ ads_seed2.atoms.coords.xf
    end

    # =========================================================================================
    # 3. IN-PLACE ALIAS & DEFAULT SOLVATION OVERLAYER
    # =========================================================================================
    @testset "Aliases & Default Overlayer" begin
        # Alias add_adsorbate!
        ads_alias_slab = add_adsorbate!(
            aspirin_slab, aspirin_xtal; n_molecules=1, n_trials=10
        )
        @test ads_alias_slab.atoms.n == n_slab_atoms + n_mol_atoms

        # Default overlayer (layer_height=3.0 Å)
        ads_default_slab = make_adsorbate_slab(aspirin_slab, aspirin_xtal; n_trials=10)
        @test ads_default_slab.atoms.n >= n_slab_atoms + n_mol_atoms
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
            aspirin_slab, aspirin_xtal; n_molecules=0
        )

        # Coordinate dimension sanity
        ads_test = make_adsorbate_slab(
            aspirin_slab, aspirin_xtal; n_molecules=1, n_trials=5
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
            aspirin_slab, aspirin_xtal; vacuum=10.0, n_trials=10
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

    # =========================================================================================
    # 6. FORCE FIELD PHYSICAL INTERACTION ENERGY VALIDATION (Water-Water H-Bonding)
    # =========================================================================================
    @testset "Intermolecular Force Field Energy Physical Validation" begin
        species = [:O, :H, :H]
        charges = [-0.8476, 0.4238, 0.4238]
        rel_w1 = [
            SVector(0.0, 0.0, 0.0), SVector(0.9572, 0.0, 0.0), SVector(-0.2399, 0.9272, 0.0)
        ]

        sigmas = [Xtals.uff_parameters(s).sigma for s in species]
        epsilons = [Xtals.uff_parameters(s).epsilon for s in species]
        radii = [Xtals.uff_parameters(s).r_cov for s in species]

        sub = Xtals.SubstrateCache(rel_w1, charges, sigmas, epsilons, radii)
        ads_template = Xtals.AdsorbateTemplate(
            species, charges, rel_w1, sigmas, epsilons, radii, [(1, 2), (1, 3)]
        )

        box_mat = [50.0 0.0 0.0; 0.0 50.0 0.0; 0.0 0.0 50.0]
        c_to_f = inv(box_mat)
        f_to_c = box_mat

        best_E = Inf
        best_r_OH = 0.0

        for r_OO in 2.2:0.05:3.8
            for trial in 1:100
                R = Xtals.random_rotation_matrix()
                w2_center = SVector(r_OO, 0.0, 0.0)
                placed_w2 = [w2_center + R * r for r in rel_w1]

                E_tot = Xtals.evaluate_interaction_energy(
                    sub, ads_template, placed_w2, c_to_f, f_to_c
                )
                if E_tot < best_E
                    best_E = E_tot
                    d_o1_h2 = norm(sub.coords[1] - placed_w2[2])
                    d_o1_h3 = norm(sub.coords[1] - placed_w2[3])
                    d_o2_h1 = norm(placed_w2[1] - sub.coords[2])
                    best_r_OH = min(d_o1_h2, d_o1_h3, d_o2_h1)
                end
            end
        end

        # Assert favorable binding energy and realistic hydrogen bond distance
        @test best_E < -10.0 # Highly favorable attractive binding energy (kcal/mol)
        @test 1.8 <= best_r_OH <= 2.4 # Realistic O...H hydrogen bond distance (Å)
    end
end
