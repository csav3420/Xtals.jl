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
    for line in hetatm_lines
        @test length(line) == 80
        @test line[1:6] == "HETATM"
        @test strip(line[18:20]) == "MOL"
        @test line[22] == 'A'
        elem = strip(line[77:78])
        @test elem in ["C", "H", "O"]
    end

    # 2. Bond inference and CONECT records
    out2 = cif_to_pdb(cif_file, temp_pdb2; infer_bonds=true, drop_cross_pb_bonds=true)
    @test isfile(temp_pdb2)
    lines2 = readlines(temp_pdb2)
    conect_lines = filter(startswith.("CONECT"), lines2)
    @test length(conect_lines) > 0

    # Verify no cross-boundary bonds exist in CONECT records
    # Read coordinates from HETATM lines
    hetatm_coords = zeros(Float64, 3, 84)
    for (idx, line) in enumerate(filter(startswith.("HETATM"), lines2))
        x = parse(Float64, line[31:38])
        y = parse(Float64, line[39:46])
        z = parse(Float64, line[47:54])
        hetatm_coords[:, idx] = [x, y, z]
    end

    for cline in conect_lines
        parts = split(cline)
        src_idx = parse(Int, parts[2])
        for tok in parts[3:end]
            dst_idx = parse(Int, tok)
            dist = norm(hetatm_coords[:, src_idx] - hetatm_coords[:, dst_idx])
            @test dist < 3.2 # all bonds intact within the cell, no ripped cross-boundary bonds
        end
    end

    # 3. Crystal object input to cif_to_pdb & write_pdb
    xtal = Crystal("aspirin.cif")
    out3 = cif_to_pdb(xtal, temp_pdb3; res_name="ASP", chain=('B'))
    @test isfile(temp_pdb3)
    lines3 = readlines(temp_pdb3)
    het_sample = lines3[findfirst(startswith.(lines3, "HETATM"))]
    @test strip(het_sample[18:20]) == "ASP"
    @test het_sample[22] == 'B'

    # 4. write_crystal dispatch to write_pdb
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
    for nl in ni_lines
        @test nl[13:14] == "Ni"
        @test strip(nl[77:78]) == "NI"
    end

    # Cleanup
    rm(temp_pdb1; force=true)
    rm(temp_pdb2; force=true)
    rm(temp_pdb3; force=true)
    rm(temp_pdb4; force=true)
    rm(temp_nipy; force=true)
end
