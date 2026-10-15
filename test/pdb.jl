using Xtals, Test, LinearAlgebra, Graphs, MetaGraphs

@testset "Lightweight CIF to PDB Converter & Native PDB Writer" begin
    set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

    cif_file = joinpath(rc[:paths][:crystals], "aspirin.cif")
    temp_pdb1 = tempname() * ".pdb"
    temp_pdb2 = tempname() * ".pdb"
    temp_pdb3 = tempname() * ".pdb"
    temp_pdb4 = tempname() * ".pdb"

    # 1. Direct cif_to_pdb conversion
    out1 = cif_to_pdb(cif_file, temp_pdb1)
    @test isfile(temp_pdb1)
    @test out1 == temp_pdb1

    lines1 = readlines(temp_pdb1)
    @test any(startswith.(lines1, "REMARK"))
    @test any(startswith.(lines1, "CRYST1"))
    @test any(startswith.(lines1, "HETATM"))
    @test any(startswith.(lines1, "END"))

    # Test CRYST1 format
    cryst_line = lines1[findfirst(startswith.(lines1, "CRYST1"))]
    @test length(cryst_line) >= 54
    @test occursin("12.270", cryst_line) # a
    @test occursin("6.558", cryst_line)  # b
    @test occursin("11.496", cryst_line) # c

    # Test HETATM lines are exactly 80 chars and properly aligned
    hetatm_lines = filter(startswith.("HETATM"), lines1)
    @test length(hetatm_lines) == 84 # 84 atoms in aspirin supercell
    @test all(length.(hetatm_lines) .== 80)
    @test all(l -> l[1:6] == "HETATM", hetatm_lines)
    @test all(l -> strip(l[18:20]) == "MOL", hetatm_lines)
    @test all(l -> l[22] == 'A', hetatm_lines)
    @test all(l -> strip(l[77:78]) in ["C", "H", "O"], hetatm_lines)

    # 2. Bond inference and CONECT records
    out2 = cif_to_pdb(cif_file, temp_pdb2; infer_bonds=true, drop_cross_pb_bonds=true)
    @test isfile(temp_pdb2)
    lines2 = readlines(temp_pdb2)
    conect_lines = filter(startswith.("CONECT"), lines2)
    @test length(conect_lines) > 0

    # Verify no cross-boundary bonds exist in CONECT records
    hetatm_coords = zeros(Float64, 3, 84)
    for (idx, line) in enumerate(filter(startswith.("HETATM"), lines2))
        x = parse(Float64, line[31:38])
        y = parse(Float64, line[39:46])
        z = parse(Float64, line[47:54])
        hetatm_coords[:, idx] = [x, y, z]
    end

    conect_dists = Float64[]
    for cline in conect_lines
        parts = split(cline)
        src_idx = parse(Int, parts[2])
        for tok in parts[3:end]
            dst_idx = parse(Int, tok)
            push!(conect_dists, norm(hetatm_coords[:, src_idx] - hetatm_coords[:, dst_idx]))
        end
    end
    @test all(conect_dists .< 3.2) # all bonds intact within the cell, no ripped cross-boundary bonds

    # 3. Crystal object input to write_crystal_pdb & cif_to_pdb
    xtal = Crystal("aspirin.cif")
    out3 = write_crystal_pdb(xtal, temp_pdb3; res_name="ASP", chain=('B'))
    @test isfile(temp_pdb3)
    @test out3 == temp_pdb3
    lines3 = readlines(temp_pdb3)
    het_sample = lines3[findfirst(startswith.(lines3, "HETATM"))]
    @test strip(het_sample[18:20]) == "ASP"
    @test het_sample[22] == 'B'

    # Unexported alias backward-compatibility test
    @test Xtals.write_pdb === write_crystal_pdb

    # 4. write_crystal dispatch to write_crystal_pdb
    write_crystal(xtal, temp_pdb4)
    @test isfile(temp_pdb4)
    lines4 = readlines(temp_pdb4)
    @test any(startswith.(lines4, "CRYST1"))
    @test any(startswith.(lines4, "HETATM"))

    # 5. 2-character element symbol formatting (e.g. Ni)
    nipy_cif = joinpath(rc[:paths][:crystals], "NiPyC2_relax.cif")
    temp_nipy = tempname() * ".pdb"
    cif_to_pdb(nipy_cif, temp_nipy)
    @test isfile(temp_nipy)
    nipy_lines = readlines(temp_nipy)
    ni_lines = filter(l -> startswith(l, "HETATM") && strip(l[77:78]) == "NI", nipy_lines)
    @test length(ni_lines) > 0
    # For Ni (2-character element), atom name starts in col 13
    @test all(nl[13:14] == "Ni" && strip(nl[77:78]) == "NI" for nl in ni_lines)

    # 6. write_water_pdb canonical water generation
    temp_wat = tempname() * ".pdb"
    write_water_pdb(temp_wat)
    @test isfile(temp_wat)
    wat_lines = readlines(temp_wat)
    @test any(startswith.(wat_lines, "REMARK"))
    wat_het = filter(startswith.("HETATM"), wat_lines)
    @test length(wat_het) == 3
    # Check atom names and order: O, H1, H2
    @test wat_het[1][13:16] == " O  "
    @test wat_het[2][13:16] == " H1 "
    @test wat_het[3][13:16] == " H2 "
    @test strip(wat_het[1][18:20]) == "HOH"
    @test wat_het[1][22] == 'A'
    # Parse coordinates and verify bond lengths and angle
    o_xyz = [
        parse(Float64, wat_het[1][31:38]),
        parse(Float64, wat_het[1][39:46]),
        parse(Float64, wat_het[1][47:54]),
    ]
    h1_xyz = [
        parse(Float64, wat_het[2][31:38]),
        parse(Float64, wat_het[2][39:46]),
        parse(Float64, wat_het[2][47:54]),
    ]
    h2_xyz = [
        parse(Float64, wat_het[3][31:38]),
        parse(Float64, wat_het[3][39:46]),
        parse(Float64, wat_het[3][47:54]),
    ]
    d_oh1 = norm(h1_xyz - o_xyz)
    d_oh2 = norm(h2_xyz - o_xyz)
    @test isapprox(d_oh1, 0.957; atol=0.01)
    @test isapprox(d_oh2, 0.957; atol=0.01)
    v1 = h1_xyz - o_xyz
    v2 = h2_xyz - o_xyz
    angle_deg = acos(dot(v1, v2) / (norm(v1) * norm(v2))) * 180.0 / π
    @test isapprox(angle_deg, 104.52; atol=0.5)
    wat_conect = filter(startswith.("CONECT"), wat_lines)
    @test length(wat_conect) == 3
    @test wat_lines[end] == "END"

    # 7. write_crystal_pdb whole-molecule wrapping & per-molecule residue sequence & atom names
    temp_asp_pdb = tempname() * ".pdb"
    write_crystal_pdb(xtal, temp_asp_pdb; res_name="ASP", whole_molecules=true)
    @test isfile(temp_asp_pdb)
    asp_lines = readlines(temp_asp_pdb)
    # Line 1 must strictly be CRYST1
    @test startswith(asp_lines[1], "CRYST1")
    asp_het = filter(startswith.("HETATM"), asp_lines)
    @test length(asp_het) == 84 # 4 molecules of 21 atoms
    @test all(length.(asp_het) .== 80)
    @test all(strip(l[18:20]) == "ASP" for l in asp_het)
    # Verify sequential residue numbering: 1, 2, 3, 4
    for m in 1:4
        m_block = asp_het[(((m - 1) * 21) + 1):(m * 21)]
        @test all(parse(Int, l[23:26]) == m for l in m_block)
        # Verify atom names in each residue are C1..C9, O1..O4, H1..H8 (unique)
        names = [strip(l[13:16]) for l in m_block]
        @test length(unique(names)) == 21
        # Check max intramolecular distance is small (molecule is whole and intact)
        coords = [parse.(Float64, [l[31:38], l[39:46], l[47:54]]) for l in m_block]
        max_dist = maximum([norm(coords[i] - coords[j]) for i in 1:21 for j in (i + 1):21])
        @test max_dist < 10.0 # Intact aspirin molecule (~6.5 Å), no cross-PBC stretching
    end

    # 8. standardize_md_pdb: broken molecule healing, water reordering, and line 1 CRYST1 restoration
    synth_pdb = tempname() * ".pdb"
    synth_out = tempname() * ".pdb"
    open(synth_pdb, "w") do f
        println(f, "REMARK Packmol generated")
        # ASP mol 1 (intact)
        for at in 1:21
            println(
                f,
                "HETATM$(lpad(at, 5)) C$(rpad(at, 3)) MOL A   1       5.000   5.000   5.000  1.00  0.00           C  ",
            )
        end
        # ASP mol 2 (atom 1 at x=0.2, atoms 2..21 at x=99.2 in a 100.0 Å box -> broken across boundary)
        println(
            f,
            "HETATM   22 C1   MOL A   1       0.200  10.000  10.000  1.00  0.00           C  ",
        )
        for at in 2:21
            println(
                f,
                "HETATM$(lpad(21 + at, 5)) C$(rpad(at, 3)) MOL A   1      99.200  10.000  10.000  1.00  0.00           C  ",
            )
        end
        # Water with inverted order H, H, O
        println(
            f,
            "HETATM   43  H   HOH A   1      20.000  20.000  20.957  1.00  0.00           H  ",
        )
        println(
            f,
            "HETATM   44  H   HOH A   1      20.000  20.927  19.760  1.00  0.00           H  ",
        )
        println(
            f,
            "HETATM   45  O   HOH A   1      20.000  20.000  20.000  1.00  0.00           O  ",
        )
        println(f, "END")
    end

    box_synth = (100.0, 100.0, 100.0)
    standardize_md_pdb(
        synth_pdb,
        synth_out;
        box=box_synth,
        crystal_res_name="ASP",
        crystal_atoms_per_mol=21,
        solvent_res_name="HOH",
        solvent_atoms_per_mol=3,
    )
    @test isfile(synth_out)
    std_lines = readlines(synth_out)
    @test startswith(std_lines[1], "CRYST1  100.000  100.000  100.000")
    std_het = filter(startswith.("HETATM"), std_lines)
    @test length(std_het) == 45
    # Residue 1: ASP 1
    @test strip(std_het[1][18:20]) == "ASP"
    @test parse(Int, std_het[1][23:26]) == 1
    # Residue 2: ASP 2
    @test strip(std_het[22][18:20]) == "ASP"
    @test parse(Int, std_het[22][23:26]) == 2
    # Verify split molecule 2 is healed (min_image unwrapped relative to atom 1):
    c1 = parse.(Float64, [std_het[22][31:38], std_het[22][39:46], std_het[22][47:54]])
    c2 = parse.(Float64, [std_het[23][31:38], std_het[23][39:46], std_het[23][47:54]])
    @test norm(c1 - c2) < 2.0 # Healed from ~99 Å across box to true ~1.2 Å bond
    # Residue 3: HOH 3
    @test strip(std_het[43][18:20]) == "HOH"
    @test parse(Int, std_het[43][23:26]) == 3
    # Water atoms reordered to O, H1, H2
    @test std_het[43][13:16] == " O  "
    @test std_het[44][13:16] == " H1 "
    @test std_het[45][13:16] == " H2 "
    @test std_lines[end] == "END"

    # Cleanup
    rm(temp_pdb1; force=true)
    rm(temp_pdb2; force=true)
    rm(temp_pdb3; force=true)
    rm(temp_pdb4; force=true)
    rm(temp_nipy; force=true)
    rm(temp_wat; force=true)
    rm(temp_asp_pdb; force=true)
    rm(synth_pdb; force=true)
    rm(synth_out; force=true)
end
