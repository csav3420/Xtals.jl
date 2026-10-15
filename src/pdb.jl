# =========================================================================================
# LIGHTWEIGHT PDB WRITER & CIF-TO-PDB CONVERTER (INDEPENDENT OF CHEMFILES)
# =========================================================================================

"""
    format_pdb_atom_name(species::Symbol, idx::Int) -> String

Format a 4-character atom name for PDB records (columns 13-16) conforming to the PDB specification:
- 1-letter elements (C, H, O, N, etc.) start in column 14 (preceded by a space).
- 2-letter elements (Fe, Cu, Cl, Si, etc.) start in column 13.
"""
function format_pdb_atom_name(species::Symbol, idx::Int)
    elem = string(species)
    name = elem * string(idx)
    if length(elem) == 1
        if length(name) == 1
            return " " * name * "  "
        elseif length(name) == 2
            return " " * name * " "
        elseif length(name) == 3
            return " " * name
        else
            return name[1:min(4, end)]
        end
    else
        if length(name) == 2
            return name * "  "
        elseif length(name) == 3
            return name * " "
        else
            return name[1:min(4, end)]
        end
    end
end

function format_pdb_atom_name(name::AbstractString)
    s = string(name)
    if length(s) == 1
        return " " * s * "  "
    elseif length(s) == 2
        first_c = uppercase(s[1])
        if first_c in ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'I', 'B']
            return " " * s * " "
        else
            return s * "  "
        end
    elseif length(s) == 3
        first_c = uppercase(s[1])
        if first_c in ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'I', 'B']
            return " " * s
        else
            return s * " "
        end
    else
        return s[1:min(4, end)]
    end
end
format_pdb_atom_name(name::Symbol) = format_pdb_atom_name(string(name))

"""
    write_crystal_pdb(crystal::Crystal, filename::AbstractString=splitext(isempty(crystal.name) ? "crystal" : crystal.name)[1] * ".pdb";
                      wrap_coords::Bool=true,
                      whole_molecules::Bool=true,
                      drop_cross_pb_bonds::Bool=true,
                      write_bonds::Bool=true,
                      res_name::Union{Nothing, AbstractString}=nothing,
                      chain::Char='A',
                      occupancy::Float64=1.0,
                      temp_factor::Float64=0.0,
                      start_res_id::Int=1,
                      start_atom_id::Int=1) -> String

Export an Xtals `Crystal` structure directly to a standard Protein Data Bank (.pdb) file
without any dependency on Chemfiles.

Features:
- CRYST1 unit cell record strictly placed as line 1 for immediate OpenMM / MD reader compatibility.
- Whole-molecule wrapping: preserves 100% of covalent bonds intact without artificial PBC stretching.
- Per-molecule residue numbering: each molecule receives a unique sequential residue sequence ID.
- Per-molecule canonical atom names: atom indices reset within each molecule (e.g. C1..C9, O1..O4, H1..H8).
- CONECT records for bonds, with automatic removal of cross-boundary periodic bonds.
"""
function write_crystal_pdb(
    crystal::Crystal,
    filename::AbstractString=splitext(isempty(crystal.name) ? "crystal" : crystal.name)[1] *
                             ".pdb";
    wrap_coords::Bool=true,
    whole_molecules::Bool=true,
    drop_cross_pb_bonds::Bool=true,
    write_bonds::Bool=true,
    res_name::Union{Nothing, AbstractString}=nothing,
    chain::Char=('A'),
    occupancy::Float64=1.0,
    temp_factor::Float64=0.0,
    start_res_id::Int=1,
    start_atom_id::Int=1,
)
    out_filename = endswith(lowercase(filename), ".pdb") ? filename : filename * ".pdb"

    c_cart = Cart(crystal.atoms.coords, crystal.box).x
    f_to_c = crystal.box.f_to_c
    c_to_f = crystal.box.c_to_f

    # Determine connected components (molecules)
    c_bonds = crystal.bonds
    if whole_molecules && ne(c_bonds) == 0 && crystal.atoms.n > 0 && crystal.atoms.n <= 5000
        try
            c_copy = deepcopy(crystal)
            infer_bonds!(c_copy, true)
            c_bonds = c_copy.bonds
        catch
        end
    end

    comps = if ne(c_bonds) > 0
        connected_components(c_bonds)
    else
        [[i] for i in 1:(crystal.atoms.n)]
    end

    # Whole-molecule or atom wrapping
    if wrap_coords
        if whole_molecules && ne(c_bonds) > 0
            # Step 1: BFS minimum-image un-wrapping within each molecule
            visited = fill(false, crystal.atoms.n)
            for comp in comps
                length(comp) <= 1 && continue
                queue = [comp[1]]
                visited[comp[1]] = true
                while !isempty(queue)
                    curr = popfirst!(queue)
                    for nbr in neighbors(c_bonds, curr)
                        if !visited[nbr]
                            visited[nbr] = true
                            d_cart = c_cart[:, nbr] - c_cart[:, curr]
                            d_frac = c_to_f * d_cart
                            d_frac .-= round.(d_frac)
                            c_cart[:, nbr] = c_cart[:, curr] + f_to_c * d_frac
                            push!(queue, nbr)
                        end
                    end
                end
            end

            # Step 2: Rigid-body translation so molecule anchor is in [0, 1)^3
            for comp in comps
                anchor_frac = c_to_f * c_cart[:, comp[1]]
                shift_frac = floor.(anchor_frac)
                shift_cart = f_to_c * shift_frac
                for idx in comp
                    c_cart[:, idx] .-= shift_cart
                end
            end
        else
            xf = c_to_f * c_cart
            xf .= mod.(xf, 1.0)
            c_cart = f_to_c * xf
        end
    end

    # Default residue name
    rname = if res_name !== nothing
        res_name
    elseif !isempty(crystal.name)
        clean_name = replace(crystal.name, r"[^A-Za-z0-9]" => "")
        isempty(clean_name) ? "MOL" : uppercase(clean_name[1:min(3, end)])
    else
        "MOL"
    end
    res_3 = rpad(rname[1:min(3, end)], 3)

    open(out_filename, "w") do f
        # CRYST1 record strictly as line 1
        if crystal.box.a > 0 && crystal.box.b > 0 && crystal.box.c > 0
            sg = if crystal.symmetry.is_p1 || isempty(crystal.symmetry.space_group)
                "P 1"
            else
                crystal.symmetry.space_group
            end
            @printf(
                f,
                "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-11s%4d\n",
                crystal.box.a,
                crystal.box.b,
                crystal.box.c,
                crystal.box.α * 180.0 / π,
                crystal.box.β * 180.0 / π,
                crystal.box.γ * 180.0 / π,
                sg[1:min(11, end)],
                1
            )
        end

        @printf(f, "REMARK   Generated by Xtals.jl (Pure Julia PDB Writer)\n")

        # HETATM records with per-molecule sequential residue numbering
        atom_serial = start_atom_id - 1
        for (m_idx, comp) in enumerate(comps)
            res_seq = mod1(start_res_id + m_idx - 1, 9999)
            local_species_counts = Dict{Symbol, Int}()
            for atom_idx in comp
                atom_serial += 1
                sp = crystal.atoms.species[atom_idx]
                local_species_counts[sp] = get(local_species_counts, sp, 0) + 1
                atom_elem = parse_element_symbol(sp)
                name_4 = format_pdb_atom_name(atom_elem, local_species_counts[sp])
                elem_right = lpad(uppercase(string(atom_elem)[1:min(2, end)]), 2)

                @printf(
                    f,
                    "HETATM%5d %-4s %-3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n",
                    mod1(atom_serial, 99999),
                    name_4,
                    res_3,
                    chain,
                    res_seq,
                    c_cart[1, atom_idx],
                    c_cart[2, atom_idx],
                    c_cart[3, atom_idx],
                    occupancy,
                    temp_factor,
                    elem_right
                )
            end
        end

        # CONECT records
        if write_bonds && ne(crystal.bonds) > 0
            max_d = min(3.2, 0.45 * minimum([crystal.box.a, crystal.box.b, crystal.box.c]))
            for i in 1:(crystal.atoms.n)
                nbrs = Int[]
                for nb in neighbors(crystal.bonds, i)
                    is_cross = false
                    if drop_cross_pb_bonds
                        if has_prop(crystal.bonds, i, nb, :cross_boundary)
                            is_cross = get_prop(crystal.bonds, i, nb, :cross_boundary)
                        else
                            d = norm(c_cart[:, i] - c_cart[:, nb])
                            is_cross = d > max_d
                        end
                    end
                    if !is_cross
                        push!(nbrs, nb)
                    end
                end

                for chunk in Iterators.partition(nbrs, 4)
                    @printf(f, "CONECT%5d", mod1(i, 99999))
                    for nb in chunk
                        @printf(f, "%5d", mod1(nb, 99999))
                    end
                    @printf(f, "\n")
                end
            end
        end

        @printf(f, "END\n")
    end

    return out_filename
end

"""
    cif_to_pdb(cif_filepath::AbstractString, pdb_filepath::Union{Nothing, AbstractString}=nothing;
               check_neutrality::Bool=false,
               check_overlap::Bool=false,
               convert_to_p1::Bool=true,
               infer_bonds::Bool=false,
               periodic_boundaries::Union{Bool, Missing}=missing,
               wrap_coords::Bool=true,
               drop_cross_pb_bonds::Bool=true,
               write_bonds::Bool=true,
               res_name::AbstractString="MOL",
               chain::Char='A',
               kwargs...) -> String

Lightweight converter that reads a Crystallographic Information File (.cif) and exports
it directly to a Protein Data Bank (.pdb) structure file without any Chemfiles dependency.

# Arguments
- `cif_filepath::AbstractString`: Path to input CIF file.
- `pdb_filepath::Union{Nothing, AbstractString}`: Path to output PDB file. Defaults to `cif_filepath` with `.pdb` extension.
- `check_neutrality::Bool`: Enforce crystal charge neutrality (default: `false` for maximum compatibility).
- `check_overlap::Bool`: Check for overlapping atoms (default: `false`).
- `convert_to_p1::Bool`: Convert space group symmetry to P1 (default: `true`).
- `infer_bonds::Bool`: Infer bonding topology if not present in the CIF (default: `false`).
- `periodic_boundaries::Union{Bool, Missing}`: Apply periodic boundaries during bond inference (default: `true` if `infer_bonds=true`).
- `wrap_coords::Bool`: Wrap fractional coordinates into the primary unit cell [0, 1)^3 (default: `true`).
- `drop_cross_pb_bonds::Bool`: Omit cross-boundary periodic bonds in CONECT records to prevent visualization tearing (default: `true`).
- `write_bonds::Bool`: Write CONECT records if bonds exist (default: `true`).
- `res_name::AbstractString`: 3-character residue name (default: `"MOL"`).
- `chain::Char`: Single-character chain ID (default: `'A'`).

# Returns
- `filename::String`: The path to the generated PDB file.
"""
function cif_to_pdb(
    cif_filepath::AbstractString,
    pdb_filepath::Union{Nothing, AbstractString}=nothing;
    check_neutrality::Bool=false,
    check_overlap::Bool=false,
    convert_to_p1::Bool=true,
    infer_bonds::Bool=false,
    periodic_boundaries::Union{Bool, Missing}=missing,
    wrap_coords::Bool=true,
    drop_cross_pb_bonds::Bool=true,
    write_bonds::Bool=true,
    res_name::AbstractString="MOL",
    chain::Char=('A'),
    kwargs...,
)
    out_pdb = if pdb_filepath === nothing
        base = splitext(cif_filepath)[1]
        base * ".pdb"
    else
        endswith(lowercase(pdb_filepath), ".pdb") ? pdb_filepath : pdb_filepath * ".pdb"
    end

    input_file = if isfile(cif_filepath)
        abspath(cif_filepath)
    elseif haskey(rc[:paths], :crystals) &&
        isfile(joinpath(rc[:paths][:crystals], cif_filepath))
        joinpath(rc[:paths][:crystals], cif_filepath)
    else
        cif_filepath
    end

    pb = infer_bonds && ismissing(periodic_boundaries) ? true : periodic_boundaries

    crystal = Crystal(
        input_file;
        check_neutrality=check_neutrality,
        check_overlap=check_overlap,
        convert_to_p1=convert_to_p1,
        infer_bonds=infer_bonds,
        periodic_boundaries=pb,
        absolute_path=true,
        kwargs...,
    )

    return write_crystal_pdb(
        crystal,
        out_pdb;
        wrap_coords=wrap_coords,
        drop_cross_pb_bonds=drop_cross_pb_bonds,
        write_bonds=write_bonds,
        res_name=res_name,
        chain=chain,
    )
end

function cif_to_pdb(
    crystal::Crystal,
    pdb_filepath::AbstractString=splitext(
        isempty(crystal.name) ? "crystal" : crystal.name
    )[1] * ".pdb";
    kwargs...,
)
    return write_crystal_pdb(crystal, pdb_filepath; kwargs...)
end

"""
    write_water_pdb(filename::AbstractString="water.pdb";
                    res_name::AbstractString="HOH",
                    chain::Char='A',
                    model::Symbol=:tip3p) -> String

Write a canonical single water molecule PDB file with standard atom ordering (`O`, `H1`, `H2`),
standard residue naming, and accurate geometry (r_OH = 0.9572 Å, angle = 104.52°).
Directly compatible with Packmol and MD engines (OpenMM, AMBER, CHARMM, GROMACS).
"""
function write_water_pdb(
    filename::AbstractString="water.pdb";
    res_name::AbstractString="HOH",
    chain::Char=('A'),
    model::Symbol=:tip3p,
)
    out_file = endswith(lowercase(filename), ".pdb") ? filename : filename * ".pdb"
    res_3 = rpad(res_name[1:min(3, end)], 3)
    open(out_file, "w") do f
        @printf(
            f,
            "REMARK   Canonical Water Molecule for Packmol / MD Simulation (%s)\n",
            uppercase(string(model))
        )
        @printf(
            f,
            "HETATM%5d %-4s %-3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n",
            1,
            " O  ",
            res_3,
            chain,
            1,
            0.000,
            0.000,
            0.000,
            1.00,
            0.00,
            " O"
        )
        @printf(
            f,
            "HETATM%5d %-4s %-3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n",
            2,
            " H1 ",
            res_3,
            chain,
            1,
            0.957,
            0.000,
            0.000,
            1.00,
            0.00,
            " H"
        )
        @printf(
            f,
            "HETATM%5d %-4s %-3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n",
            3,
            " H2 ",
            res_3,
            chain,
            1,
            -0.240,
            0.927,
            0.000,
            1.00,
            0.00,
            " H"
        )
        @printf(f, "CONECT%5d%5d%5d\n", 1, 2, 3)
        @printf(f, "CONECT%5d%5d\n", 2, 1)
        @printf(f, "CONECT%5d%5d\n", 3, 1)
        @printf(f, "END\n")
    end
    return out_file
end

"""
    standardize_md_pdb(
        input_pdb::AbstractString,
        output_pdb::AbstractString;
        box::Union{Box, Tuple{Real, Real, Real}, Vector{<:Real}, Nothing}=nothing,
        crystal_res_name::AbstractString="ASP",
        crystal_atoms_per_mol::Int=21,
        solvent_res_name::AbstractString="HOH",
        solvent_atoms_per_mol::Int=3,
        chain::Char='A'
    ) -> String

Standardize an assembled PDB (such as a Packmol output combining a crystal slab and solvent droplet)
into a 100% MD-ready PDB file:
1. Injects the `CRYST1` record strictly on line 1 with exact simulation box dimensions.
2. Unwraps any crystal and solvent molecules broken across periodic boundaries (minimum-image convention).
3. Sequentially numbers residues: crystal molecules (1..N_cryst), then solvent molecules (N_cryst+1..N_total).
4. Standardizes canonical atom names:
   - Crystal: per-molecule unique atom names (e.g. C1..C9, O1..O4, H1..H8).
   - Solvent: canonical O, H1, H2 ordering (auto-reordering from H, H, O if needed).
5. Writes strict 80-column PDB records ending with END.
"""
function standardize_md_pdb(
    input_pdb::AbstractString,
    output_pdb::AbstractString;
    box::Union{Box, Tuple{Real, Real, Real}, Vector{<:Real}, Nothing}=nothing,
    crystal_res_name::AbstractString="ASP",
    crystal_atoms_per_mol::Int=21,
    solvent_res_name::AbstractString="HOH",
    solvent_atoms_per_mol::Int=3,
    chain::Char=('A'),
)
    lines = readlines(input_pdb)

    # Resolve box dimensions
    box_a, box_b, box_c = 0.0, 0.0, 0.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    if box isa Box
        box_a, box_b, box_c = box.a, box.b, box.c
        alpha, beta, gamma = box.α * 180.0 / π, box.β * 180.0 / π, box.γ * 180.0 / π
    elseif box isa Tuple || box isa Vector
        box_a, box_b, box_c = Float64(box[1]), Float64(box[2]), Float64(box[3])
    else
        cryst_idx = findfirst(l -> startswith(l, "CRYST1"), lines)
        if cryst_idx !== nothing
            cline = lines[cryst_idx]
            box_a = parse(Float64, cline[7:15])
            box_b = parse(Float64, cline[16:24])
            box_c = parse(Float64, cline[25:33])
            alpha = parse(Float64, cline[34:40])
            beta = parse(Float64, cline[41:47])
            gamma = parse(Float64, cline[48:54])
        else
            throw(
                ArgumentError(
                    "No CRYST1 record found in $(input_pdb). You must supply the `box` argument.",
                ),
            )
        end
    end
    box_vec = [box_a, box_b, box_c]

    atom_lines = filter(l -> startswith(l, "ATOM") || startswith(l, "HETATM"), lines)

    get_res_name(l) = length(l) >= 20 ? strip(uppercase(l[18:20])) : ""

    # Partition into crystal lines and solvent lines
    cryst_tag = uppercase(strip(crystal_res_name))
    solv_tag = uppercase(strip(solvent_res_name))

    solv_lines = filter(atom_lines) do l
        r = get_res_name(l)
        return r == solv_tag || r == "WAT" || r == "HOH" || occursin(solv_tag, r)
    end

    cryst_lines = filter(atom_lines) do l
        r = get_res_name(l)
        return r == cryst_tag || r == "MOL" || occursin(cryst_tag, r)
    end

    if isempty(cryst_lines) && isempty(solv_lines)
        n_tot = length(atom_lines)
        n_cryst_mols = div(n_tot, crystal_atoms_per_mol)
        n_cryst_atoms = n_cryst_mols * crystal_atoms_per_mol
        cryst_lines = atom_lines[1:n_cryst_atoms]
        solv_lines = atom_lines[(n_cryst_atoms + 1):end]
    elseif isempty(cryst_lines) && !isempty(solv_lines)
        solv_set = Set(solv_lines)
        cryst_lines = filter(l -> !(l in solv_set), atom_lines)
    elseif !isempty(cryst_lines) && isempty(solv_lines)
        cryst_set = Set(cryst_lines)
        solv_lines = filter(l -> !(l in cryst_set), atom_lines)
    end

    n_cryst_mols = div(length(cryst_lines), crystal_atoms_per_mol)
    n_solv_mols = div(length(solv_lines), solvent_atoms_per_mol)

    parse_xyz(l) =
        [parse(Float64, l[31:38]), parse(Float64, l[39:46]), parse(Float64, l[47:54])]

    function min_image(pos, ref, box_v)
        d = pos - ref
        d[1] -= box_v[1] * round(d[1] / box_v[1])
        d[2] -= box_v[2] * round(d[2] / box_v[2])
        d[3] -= box_v[3] * round(d[3] / box_v[3])
        return ref + d
    end

    asp_default_names = [
        "C1",
        "C2",
        "C3",
        "H1",
        "C4",
        "H2",
        "C5",
        "H3",
        "C6",
        "H4",
        "C7",
        "O1",
        "O2",
        "H5",
        "O3",
        "C8",
        "O4",
        "C9",
        "H6",
        "H7",
        "H8",
    ]

    out_filename =
        endswith(lowercase(output_pdb), ".pdb") ? output_pdb : output_pdb * ".pdb"
    open(out_filename, "w") do io
        # Line 1: CRYST1
        @printf(
            io,
            "CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1\n",
            box_a,
            box_b,
            box_c,
            alpha,
            beta,
            gamma
        )
        @printf(io, "REMARK   Clean MD-Ready Structure generated by Xtals.jl\n")

        atom_serial = 1
        cryst_res_3 = rpad(crystal_res_name[1:min(3, end)], 3)

        # 1. Write Crystal molecules
        for mol_i in 1:n_cryst_mols
            res_id = mod1(mol_i, 9999)
            mol_block = cryst_lines[(((mol_i - 1) * crystal_atoms_per_mol) + 1):(mol_i * crystal_atoms_per_mol)]
            mol_coords = [parse_xyz(l) for l in mol_block]

            anchor = mol_coords[1]
            for at_i in 2:crystal_atoms_per_mol
                mol_coords[at_i] = min_image(mol_coords[at_i], anchor, box_vec)
            end

            # Rigid shift if anchor is outside [0, box_vec)
            shift = floor.(anchor ./ box_vec) .* box_vec
            if any(shift .!= 0.0)
                for at_i in 1:crystal_atoms_per_mol
                    mol_coords[at_i] .-= shift
                end
            end

            for at_i in 1:crystal_atoms_per_mol
                c = mol_coords[at_i]
                raw_name = strip(mol_block[at_i][13:16])
                name =
                    if (cryst_tag == "ASP" || occursin("ASP", cryst_tag)) &&
                        crystal_atoms_per_mol == 21 &&
                        at_i <= length(asp_default_names)
                        asp_default_names[at_i]
                    elseif !isempty(raw_name)
                        raw_name
                    else
                        "A$(at_i)"
                    end
                elem_sym = parse_element_symbol(Symbol(name))
                name_4 = format_pdb_atom_name(name)
                elem_right = lpad(uppercase(string(elem_sym)[1:min(2, end)]), 2)
                @printf(
                    io,
                    "HETATM%5d %-4s %-3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n",
                    mod1(atom_serial, 99999),
                    name_4,
                    cryst_res_3,
                    chain,
                    res_id,
                    c[1],
                    c[2],
                    c[3],
                    1.0,
                    0.0,
                    elem_right
                )
                atom_serial += 1
            end
        end

        # 2. Write Solvent molecules
        solv_res_3 = rpad(solvent_res_name[1:min(3, end)], 3)
        for wat_i in 1:n_solv_mols
            res_id = mod1(n_cryst_mols + wat_i, 9999)
            wat_block = solv_lines[(((wat_i - 1) * solvent_atoms_per_mol) + 1):(wat_i * solvent_atoms_per_mol)]
            wat_coords = [parse_xyz(l) for l in wat_block]
            raw_names = [strip(l[13:16]) for l in wat_block]

            o_idx = findfirst(n -> startswith(uppercase(n), "O"), raw_names)
            if o_idx === nothing
                o_idx = 1
            end

            ordered_coords = if o_idx == 3
                [wat_coords[3], wat_coords[1], wat_coords[2]]
            elseif o_idx == 1
                [wat_coords[1], wat_coords[2], wat_coords[3]]
            else
                [wat_coords[o_idx], wat_coords[1], wat_coords[3]]
            end

            o_anchor = ordered_coords[1]
            ordered_coords[2] = min_image(ordered_coords[2], o_anchor, box_vec)
            ordered_coords[3] = min_image(ordered_coords[3], o_anchor, box_vec)

            # Rigid shift if water oxygen is outside [0, box_vec)
            shift = floor.(o_anchor ./ box_vec) .* box_vec
            if any(shift .!= 0.0)
                for at_i in 1:3
                    ordered_coords[at_i] .-= shift
                end
            end

            wat_names = ["O", "H1", "H2"]
            wat_elems = ["O", "H", "H"]
            for at_i in 1:3
                c = ordered_coords[at_i]
                name_4 = format_pdb_atom_name(wat_names[at_i])
                elem_right = lpad(wat_elems[at_i], 2)
                @printf(
                    io,
                    "HETATM%5d %-4s %-3s %c%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  \n",
                    mod1(atom_serial, 99999),
                    name_4,
                    solv_res_3,
                    chain,
                    res_id,
                    c[1],
                    c[2],
                    c[3],
                    1.0,
                    0.0,
                    elem_right
                )
                atom_serial += 1
            end
        end

        @printf(io, "END\n")
    end
    return out_filename
end

# Backward-compatibility alias (unexported to avoid PDBTools collision)
const write_pdb = write_crystal_pdb
