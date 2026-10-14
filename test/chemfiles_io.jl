using Test
using Xtals
using LinearAlgebra

set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

@testset "Chemfiles I/O, Multi-Format Readers & Adsorbate Unit Tests" begin
    aspirin_xtal = Crystal("aspirin.cif")
    aspirin_slab = make_slab(aspirin_xtal, (1, 1, 0); vacuum=10.0, layers=1)[1]
    n_slab_atoms = aspirin_slab.atoms.n

    # =========================================================================================
    # 1. CHEMFILES BRIDGE & EXPORTER TESTS (Pruned & Lightweight)
    # =========================================================================================
    @testset "Chemfiles Bridge & Universal Exporter" begin
        # 1. Read-in bridge check
        xtal_cif = read_adsorbate(joinpath(rc[:paths][:crystals], "aspirin.cif"))
        @test xtal_cif isa Crystal
        @test xtal_cif.atoms.n > 0

        # 2. Element Symbol Sanitization Tests (PeriodicTable.jl Integration)
        @test Xtals.parse_element_symbol("OH") == :O
        @test Xtals.parse_element_symbol("OW") == :O
        @test Xtals.parse_element_symbol("HW1") == :H
        @test Xtals.parse_element_symbol("C12") == :C
        @test Xtals.parse_element_symbol("Fe_a") == :Fe
        @test Xtals.parse_element_symbol("X", 8) == :O
        @test Xtals.parse_element_symbol("X", 1) == :H

        # 3. File missing error guard
        @test_throws ErrorException read_adsorbate("non_existent_adsorbate.xyz")

        # 3. Exporter bridge check (write & re-read)
        tmp_xyz = joinpath(tempdir(), "test_export_aspirin.xyz")
        try
            write_crystal(aspirin_xtal, tmp_xyz)
            @test isfile(tmp_xyz)
            xtal_re = read_crystal_chemfiles(tmp_xyz)
            @test xtal_re.atoms.n == aspirin_xtal.atoms.n
            @test xtal_re.atoms.species == aspirin_xtal.atoms.species
        finally
            rm(tmp_xyz; force=true)
        end
    end

    # =========================================================================================
    # 2. SINGLE-MOLECULE UNIT EXTRACTION & DEPOSITION SCALING
    # =========================================================================================
    @testset "Single-Molecule Adsorbate Unit & Deposition Scaling" begin
        ads_cif = read_adsorbate(joinpath(rc[:paths][:crystals], "aspirin.cif"))

        # detect_molecules extracts individual molecular units
        mols = detect_molecules(ads_cif)
        @test !isempty(mols)
        n_mol_atoms = length(mols[1].species)
        @test n_mol_atoms > 0

        # prepare_adsorbate_template extracts exactly 1 molecule template
        template = Xtals.prepare_adsorbate_template(ads_cif)
        @test length(template.species) == n_mol_atoms

        # Deposition scaling: 1 molecule deposition
        ads_deposited_slab = make_adsorbate_slab(
            aspirin_slab,
            joinpath(rc[:paths][:crystals], "aspirin.cif");
            n_molecules=1,
            n_trials=5,
        )
        @test ads_deposited_slab.atoms.n == n_slab_atoms + n_mol_atoms

        # 3 molecules deposition scaling
        ads_3mol_slab = make_adsorbate_slab(
            aspirin_slab,
            joinpath(rc[:paths][:crystals], "aspirin.cif");
            n_molecules=3,
            n_trials=5,
        )
        @test ads_3mol_slab.atoms.n == n_slab_atoms + (3 * n_mol_atoms)

        # Zero atom overlaps assertion
        overlap_flag, overlap_pairs = overlap(ads_deposited_slab, false)
        @test !overlap_flag
        @test isempty(overlap_pairs)
    end

    # =========================================================================================
    # 3. QUATERNION ROTATION MATRIX INVARIANCE (Aggregated into 3 Assertions)
    # =========================================================================================
    @testset "Quaternion Rotation Matrix Invariance" begin
        # Generate and store 100 rotation matrices
        rotations = [Xtals.random_rotation_matrix() for _ in 1:100]

        # 1. Vectorized Orthogonality Check (R^T * R == I_3)
        @test all(R' * R ≈ I(3) for R in rotations)

        # 2. Vectorized Determinant Check (det(R) == 1.0)
        @test all(det(R) ≈ 1.0 for R in rotations)

        # 3. Vectorized Rigid Body Vector Norm Invariance
        vec_orig = [0.9572, 0.4238, 0.112]
        norm_orig = norm(vec_orig)
        @test all(norm(R * vec_orig) ≈ norm_orig for R in rotations)
    end
end
