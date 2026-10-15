using Chemfiles
using Graphs
using LinearAlgebra
using PeriodicTable

# =========================================================================================
# 0. ELEMENT SYMBOL PARSER (PeriodicTable.jl Integration)
# =========================================================================================

"""
    parse_element_symbol(sym::Union{Symbol, AbstractString}, atomic_number::Int=0, fallback_name::AbstractString="") -> Symbol

Parse and sanitize raw atom names or force-field type strings (e.g. "OH", "OW", "HW1", "C12") into clean periodic table
element symbols (:O, :H, :C, :Fe) using PeriodicTable.jl so force field parameters and CIF/XYZ exporters recognize elements cleanly.
"""
function parse_element_symbol(
    sym::Union{Symbol, AbstractString},
    atomic_number::Int=0,
    fallback_name::AbstractString="",
)
    # 1. Primary: Use atomic_number Z from Chemfiles if valid (1 <= Z <= 118)
    if 1 <= atomic_number <= 118
        return Symbol(PeriodicTable.elements[atomic_number].symbol)
    end

    s = string(sym)
    if isempty(s) || s == "X"
        s = string(fallback_name)
    end

    # Remove digits, underscores, dashes, and pluses (e.g. "HW1" -> "HW", "C12" -> "C", "OH" -> "OH")
    clean_s = replace(s, r"[0-9_\-\+]" => "")
    if isempty(clean_s)
        return :X
    end

    # Fast single-character check for Hydrogen
    # In PDB/MOL2/CIF, "HO", "HO1", "HW1", "H1" start with H.
    # Only exact "Ho" (lowercase o) is Holmium (element 67) or "Hf" is Hafnium (element 72).
    if clean_s[1] == 'H' || clean_s[1] == 'h'
        if length(clean_s) >= 2 && (clean_s[1:2] == "Ho" || clean_s[1:2] == "Hf")
            return Symbol(clean_s[1:2])
        else
            return :H
        end
    end

    # Check 2-character element prefix (e.g. :Fe, :Cu, :Si, :Cl, :Br)
    if length(clean_s) >= 2
        prefix2 = Symbol(uppercasefirst(lowercase(clean_s[1:2])))
        if haskey(PeriodicTable.elements, prefix2)
            return prefix2
        end
    end

    # Check 1-character element prefix (e.g. :H, :O, :C, :N, :S, :P, :F)
    prefix1 = Symbol(uppercasefirst(lowercase(clean_s[1:1])))
    if haskey(PeriodicTable.elements, prefix1)
        return prefix1
    end

    return prefix1
end

# =========================================================================================
# 1. CHEMFILES BRIDGE & UNIVERSAL CRYSTAL READER
# =========================================================================================

"""
    read_crystal_chemfiles(filename::AbstractString; name::Union{Nothing, String}=nothing) -> Crystal

Read any computational chemistry structure file (.pdb, .mol2, .xyz, .sdf, .gro, .cif, .mmcif)
using Chemfiles.jl and convert it into an Xtals `Crystal` object.
"""
function read_crystal_chemfiles(
    filename::AbstractString; name::Union{Nothing, String}=nothing
)
    if !isfile(filename)
        error("File not found: $filename")
    end

    traj = Chemfiles.Trajectory(filename)
    frame = Chemfiles.read(traj)
    close(traj)

    n_atoms = length(frame)
    chem_topo = Chemfiles.Topology(frame)
    chem_pos = Chemfiles.positions(frame) # Matrix of 3D positions in Å (3 x N)

    species_vec = Symbol[]
    coords_cart = zeros(Float64, 3, n_atoms)

    for i in 1:n_atoms
        atom_i = Chemfiles.Atom(chem_topo, UInt64(i - 1))
        elem_type = Chemfiles.type(atom_i)
        elem_name = Chemfiles.name(atom_i)
        atomic_num = Int(Chemfiles.atomic_number(atom_i))

        clean_species = parse_element_symbol(elem_type, atomic_num, elem_name)
        push!(species_vec, clean_species)

        coords_cart[1, i] = chem_pos[1, i]
        coords_cart[2, i] = chem_pos[2, i]
        coords_cart[3, i] = chem_pos[3, i]
    end

    # Unit Cell Box determination
    chem_box = Chemfiles.UnitCell(frame)
    box_shape = Chemfiles.shape(chem_box)

    if box_shape == Chemfiles.Infinite
        # Non-periodic system: build bounding box around Cartesian coordinates
        min_xyz = vec(minimum(coords_cart; dims=2))
        max_xyz = vec(maximum(coords_cart; dims=2))
        span_xyz = max_xyz .- min_xyz
        padded_span = max.(10.0, span_xyz .+ 10.0)

        f_to_c = [
            padded_span[1] 0.0 0.0
            0.0 padded_span[2] 0.0
            0.0 0.0 padded_span[3]
        ]
        box = Box(f_to_c)
    else
        # Periodic system: extract unit cell vectors from matrix
        box_mat = Chemfiles.matrix(chem_box) # 3x3 matrix column vectors
        box = Box(box_mat)
    end

    xtal_name = name !== nothing ? name : splitext(basename(filename))[1]
    atoms_cart = Cart(coords_cart)
    atoms_frac = Frac(atoms_cart, box)
    atoms = Atoms(species_vec, atoms_frac)
    charges = Charges(zeros(Float64, n_atoms), atoms_frac)

    crystal = Crystal(xtal_name, box, atoms, charges)

    # Extract intra-molecular bonds if present in topology
    n_bonds = Chemfiles.bonds_count(chem_topo)
    if n_bonds > 0
        chem_bonds = Chemfiles.bonds(chem_topo)
        for b in 1:n_bonds
            u = Int(chem_bonds[1, b]) + 1
            v = Int(chem_bonds[2, b]) + 1
            add_edge!(crystal.bonds, u, v)
        end
    else
        # Infer bonds fallback
        infer_bonds!(crystal, false)
    end

    return crystal
end

# =========================================================================================
# 2. UNIVERSAL FILE EXPORTER (write_crystal & write_crystal_chemfiles)
# =========================================================================================

"""
    write_crystal_chemfiles(crystal::Crystal, filename::AbstractString)

Export an Xtals `Crystal` object to non-CIF file formats supported by Chemfiles (.xyz, .pdb, .mol2, .sdf, .gro).
Wraps fractional coordinates into [0, 1)^3 so atoms sit cleanly inside the cell box.
"""
function write_crystal_chemfiles(crystal::Crystal, filename::AbstractString)
    # Wrap fractional coordinates to [0, 1)^3 so atoms and molecules sit cleanly inside the cell box
    xf_wrapped = mod.(crystal.atoms.coords.xf, 1.0)
    c_cart = crystal.box.f_to_c * xf_wrapped
    n_atoms = crystal.atoms.n

    frame = Chemfiles.Frame()

    for i in 1:n_atoms
        atom_i = Chemfiles.Atom(String(crystal.atoms.species[i]))
        if has_charges(crystal)
            Chemfiles.set_charge!(atom_i, crystal.charges.q[i])
        end
        Chemfiles.add_atom!(frame, atom_i, [c_cart[1, i], c_cart[2, i], c_cart[3, i]])
    end

    # Set unit cell
    chem_box = Chemfiles.UnitCell(crystal.box.f_to_c)
    Chemfiles.set_cell!(frame, chem_box)

    # Export bonds if present
    if ne(crystal.bonds) > 0
        chem_topo = Chemfiles.Topology(frame)
        for e in edges(crystal.bonds)
            Chemfiles.add_bond!(chem_topo, UInt64(src(e) - 1), UInt64(dst(e) - 1))
        end
        Chemfiles.set_topology!(frame, chem_topo)
    end

    traj = Chemfiles.Trajectory(filename, 'w')
    Chemfiles.write(traj, frame)
    close(traj)

    return nothing
end

"""
    write_crystal(crystal::Crystal, filename::AbstractString; kwargs...)

Universal file exporter dispatch:
- Dispatches `.cif` files directly to `write_cif(crystal, filename; kwargs...)` for full crystallographic CIF export.
- Dispatches `.pdb` files directly to native `write_crystal_pdb(crystal, filename; kwargs...)` (zero Chemfiles dependency).
- Uses `write_crystal_chemfiles` for all other formats (.xyz, .mol2, .sdf, .gro).
"""
function write_crystal(crystal::Crystal, filename::AbstractString; kwargs...)
    if endswith(lowercase(filename), ".cif")
        write_cif(crystal, filename; kwargs...)
    elseif endswith(lowercase(filename), ".pdb")
        write_crystal_pdb(crystal, filename; kwargs...)
    else
        write_crystal_chemfiles(crystal, filename)
    end
end

# =========================================================================================
# 3. MOLECULAR DYNAMICS & MONTE CARLO TRAJECTORY READER
# =========================================================================================

"""
    read_trajectory(filename::AbstractString; box::Union{Nothing, Box}=nothing) -> Vector{Crystal}

Read all frames of a simulation trajectory (.dcd, .xtc, .trr, .lammpstrj, multi-frame .xyz)
into a `Vector{Crystal}`.
"""
function read_trajectory(filename::AbstractString; box::Union{Nothing, Box}=nothing)
    if !isfile(filename)
        throw(ArgumentError("Trajectory file not found: $filename"))
    end

    traj = Chemfiles.Trajectory(filename)
    n_frames = length(traj)
    frames = Vector{Crystal}(undef, n_frames)

    for step in 1:n_frames
        frame = Chemfiles.read(traj)
        n_atoms = length(frame)
        chem_topo = Chemfiles.Topology(frame)
        chem_pos = Chemfiles.positions(frame)

        species_vec = Symbol[]
        coords_cart = zeros(Float64, 3, n_atoms)

        for i in 1:n_atoms
            atom_i = Chemfiles.Atom(chem_topo, UInt64(i - 1))
            elem_str = Chemfiles.type(atom_i)
            if isempty(elem_str) || elem_str == "X"
                elem_str = Chemfiles.name(atom_i)
            end
            push!(species_vec, Symbol(elem_str))

            coords_cart[1, i] = chem_pos[1, i]
            coords_cart[2, i] = chem_pos[2, i]
            coords_cart[3, i] = chem_pos[3, i]
        end

        frame_box = box
        if frame_box === nothing
            chem_box = Chemfiles.UnitCell(frame)
            if Chemfiles.shape(chem_box) == Chemfiles.Infinite
                min_xyz = vec(minimum(coords_cart; dims=2))
                max_xyz = vec(maximum(coords_cart; dims=2))
                span_xyz = max_xyz .- min_xyz
                padded_span = max.(10.0, span_xyz .+ 10.0)
                frame_box = Box(
                    [padded_span[1] 0.0 0.0; 0.0 padded_span[2] 0.0; 0.0 0.0 padded_span[3]]
                )
            else
                frame_box = Box(Chemfiles.matrix(chem_box))
            end
        end

        atoms_cart = Cart(coords_cart)
        atoms_frac = Frac(atoms_cart, frame_box)
        atoms = Atoms(species_vec, atoms_frac)
        charges = Charges(zeros(Float64, n_atoms), atoms_frac)

        xtal_name = "$(splitext(basename(filename))[1])_frame_$(step)"
        c_frame = Crystal(xtal_name, frame_box, atoms, charges)

        n_bonds = Chemfiles.bonds_count(chem_topo)
        if n_bonds > 0
            chem_bonds = Chemfiles.bonds(chem_topo)
            for b in 1:n_bonds
                u = Int(chem_bonds[1, b]) + 1
                v = Int(chem_bonds[2, b]) + 1
                add_edge!(c_frame.bonds, u, v)
            end
        end

        frames[step] = c_frame
    end

    close(traj)
    return frames
end
