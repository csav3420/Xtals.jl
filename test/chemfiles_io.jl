using Test
using Xtals
using LinearAlgebra

set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

@testset "Chemfiles I/O, Multi-Format Readers & Adsorbate Unit Tests" begin
    aspirin_xtal = Crystal("aspirin.cif")
    aspirin_slab = make_slab(aspirin_xtal, (1, 1, 0); vacuum=10.0, layers=1)[1]
    n_slab_atoms = aspirin_slab.atoms.n

    # =========================================================================================
    # 1. MULTI-FORMAT READERS & FILE ERROR GUARDS
    # =========================================================================================
    @testset "Multi-Format File Readers & Error Guards" begin
        # CIF reading via read_crystal_chemfiles
        xtal_cif = read_crystal_chemfiles(joinpath(rc[:paths][:crystals], "aspirin.cif"))
        @test xtal_cif.atoms.n == aspirin_xtal.atoms.n

        # Non-existent file guard
        @test_throws ErrorException read_adsorbate("non_existent_adsorbate.xyz")
        @test_throws ErrorException read_crystal_chemfiles("non_existent_file.mol2")
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

        # Deposition scaling: 1 molecule at a time per site per layer
        ads_deposited_slab = make_adsorbate_slab(
            aspirin_slab,
            joinpath(rc[:paths][:crystals], "aspirin.cif");
            coverage=(1, 1),
            n_layers=1,
            n_trials=5,
        )
        @test ads_deposited_slab.atoms.n == n_slab_atoms + n_mol_atoms

        # 2x2 grid deposition scaling (4 molecules deposited)
        ads_2x2_slab = make_adsorbate_slab(
            aspirin_slab,
            joinpath(rc[:paths][:crystals], "aspirin.cif");
            coverage=(2, 2),
            n_layers=1,
            n_trials=5,
        )
        @test ads_2x2_slab.atoms.n == n_slab_atoms + (4 * n_mol_atoms)

        # Zero atom overlaps assertion
        overlap_flag, overlap_pairs = overlap(ads_deposited_slab, false)
        @test !overlap_flag
        @test isempty(overlap_pairs)
    end

    # =========================================================================================
    # 3. UNIVERSAL EXPORTER & ROUNDTRIP CHECKS (write_crystal)
    # =========================================================================================
    @testset "Universal File Exporter & Roundtrip Verification" begin
        tmp_xyz = joinpath(tempdir(), "test_export_aspirin.xyz")
        tmp_pdb = joinpath(tempdir(), "test_export_aspirin.pdb")
        tmp_mol2 = joinpath(tempdir(), "test_export_aspirin.mol2")

        try
            # Export to XYZ, PDB, MOL2
            write_crystal(aspirin_xtal, tmp_xyz)
            write_crystal(aspirin_xtal, tmp_pdb)
            write_crystal(aspirin_xtal, tmp_mol2)

            @test isfile(tmp_xyz)
            @test isfile(tmp_pdb)
            @test isfile(tmp_mol2)

            # Roundtrip re-read verification
            xtal_re_xyz = read_crystal_chemfiles(tmp_xyz)
            xtal_re_pdb = read_crystal_chemfiles(tmp_pdb)
            xtal_re_mol2 = read_crystal_chemfiles(tmp_mol2)

            @test xtal_re_xyz.atoms.n == aspirin_xtal.atoms.n
            @test xtal_re_pdb.atoms.n == aspirin_xtal.atoms.n
            @test xtal_re_mol2.atoms.n == aspirin_xtal.atoms.n

            @test xtal_re_xyz.atoms.species == aspirin_xtal.atoms.species
            @test xtal_re_pdb.atoms.species == aspirin_xtal.atoms.species
            @test xtal_re_mol2.atoms.species == aspirin_xtal.atoms.species
        finally
            # Clean up temporary test files
            rm(tmp_xyz; force=true)
            rm(tmp_pdb; force=true)
            rm(tmp_mol2; force=true)
        end
    end

    # =========================================================================================
    # 4. QUATERNION ROTATION MATRIX RIGID BODY INVARIANCE
    # =========================================================================================
    @testset "Quaternion Rotation Matrix Orthogonality & Rigid Body Invariance" begin
        for _ in 1:100
            R = Xtals.random_rotation_matrix()

            # Orthogonality assertion R^T * R == I_3
            @test R' * R ≈ I(3) atol=1e-12

            # Proper rotation determinant det(R) == 1.0
            @test det(R) ≈ 1.0 atol=1e-12
        end

        # Rigid body distance invariance under quaternion rotation
        vec_orig = [0.9572, 0.0, 0.0]
        dist_orig = norm(vec_orig)
        for _ in 1:50
            R = Xtals.random_rotation_matrix()
            vec_rot = R * vec_orig
            @test norm(vec_rot) ≈ dist_orig atol=1e-12
        end
    end
end
