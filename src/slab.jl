using LinearAlgebra
using Graphs
using StaticArrays

"""
    MoleculeData(atom_indices, species, relative_coords_cart, charges, centroid_cart, edges)

Data container holding pre-computed 3D Cartesian information and intra-molecular bonding edges
for an individual molecule extracted from a crystal structure.

# Fields
- `atom_indices::Vector{Int}`: Global atom indices of the molecule in the source Crystal.
- `species::Vector{Symbol}`: Atomic species of each atom in the molecule.
- `relative_coords_cart::Matrix{Float64}`: 3 x N matrix of Cartesian coordinates relative to the molecular centroid.
- `charges::Vector{Float64}`: Partial atomic charges of each atom.
- `centroid_cart::Vector{Float64}`: 3D Cartesian position of the molecular centroid.
- `edges::Vector{Tuple{Int, Int}}`: Local intra-molecular bond edges (pairs of 1-indexed local atom indices).
"""
struct MoleculeData
    atom_indices::Vector{Int}
    species::Vector{Symbol}
    relative_coords_cart::Matrix{Float64} # 3 x N matrix of Cartesian coords relative to centroid
    charges::Vector{Float64}
    centroid_cart::Vector{Float64}        # 3D Cartesian centroid position
    edges::Vector{Tuple{Int, Int}}        # Local atom index pairs (1..n_m, 1..n_m)
end

"""
    surface_basis(hkl::Tuple{Int, Int, Int}; f_to_c::Union{Nothing, Matrix{Float64}}=nothing) -> Matrix{Int}

Compute a 3x3 integer transformation matrix M = [m1 m2 m3] for Miller indices (h, k, l)
where m1 and m2 lie in the (h, k, l) surface plane, m3 is out-of-plane, det(M) = 1, and m1, m2
are the shortest primitive in-plane lattice vectors.

# Mathematical Formulation
Given Miller indices (h, k, l), the surface plane equation in direct space is:
    h*x + k*y + l*z = 0

1. **Primitive In-plane Vector Search**:
   Find all short integer vectors m in Z^3 satisfying h*mx + k*my + l*mz = 0 with gcd(|mx|, |my|, |mz|) = 1.
   Sort candidate vectors by real-space Cartesian length ||f_to_c * m|| (if f_to_c provided) or ||m||.
2. **Minimal Primitive Surface Basis**:
   Select m1 as the shortest primitive in-plane vector. Select m2 as the shortest in-plane vector
   linearly independent from m1 such that an out-of-plane vector m3 in Z^3 exists with det([m1 m2 m3]) = 1.
3. **Determinant Orientation**:
   If det(M) < 0, m1 is inverted to enforce a right-handed system (det(M) = 1).
"""
function surface_basis(hkl::Tuple{Int, Int, Int}; f_to_c::Union{Nothing, Matrix{Float64}}=nothing)
    h, k, l = hkl
    # Step 1: Guard against invalid (0, 0, 0) Miller indices
    if h == 0 && k == 0 && l == 0
        throw(ArgumentError("Miller indices cannot be (0, 0, 0)"))
    end

    # Step 2: Reduce Miller indices by their greatest common divisor
    g_hkl = gcd(gcd(h, k), l)
    h_c, k_c, l_c = div(h, g_hkl), div(k, g_hkl), div(l, g_hkl)

    # Step 3: Search for short coprime in-plane integer vectors m in Z^3 satisfying h_c*mx + k_c*my + l_c*mz = 0
    in_plane = Vector{Int}[]
    for mx in -5:5, my in -5:5, mz in -5:5
        if mx == 0 && my == 0 && mz == 0
            continue
        end
        if h_c * mx + k_c * my + l_c * mz == 0
            if gcd(gcd(abs(mx), abs(my)), abs(mz)) == 1
                push!(in_plane, [mx, my, mz])
            end
        end
    end

    # Sort by real-space Cartesian length if f_to_c provided, else integer norm
    if f_to_c !== nothing
        sort!(in_plane, by=m -> norm(f_to_c * m))
    else
        sort!(in_plane, by=m -> norm(m))
    end

    # Step 4: Find shortest primitive vectors m1, m2, m3 such that det([m1 m2 m3]) == 1
    m1 = in_plane[1]
    m2 = m1
    m3 = [0, 0, 0]

    candidates = Tuple{Float64, Vector{Int}, Vector{Int}}[]

    for m in in_plane
        if norm(cross(m1, m)) > 1e-5
            for m3x in -3:3, m3y in -3:3, m3z in -3:3
                M_cand = [m1 m [m3x, m3y, m3z]]
                d = round(Int, det(M_cand))
                if d == 1 || d == -1
                    m3_vec = d < 0 ? -[m3x, m3y, m3z] : [m3x, m3y, m3z]
                    m3_cart = (f_to_c !== nothing) ? (f_to_c * m3_vec) : Float64.(m3_vec)
                    m2_cart = (f_to_c !== nothing) ? (f_to_c * m) : Float64.(m)
                    cost = norm(m2_cart) + norm(m3_cart)
                    push!(candidates, (cost, m, m3_vec))
                end
            end
        end
    end

    if !isempty(candidates)
        sort!(candidates, by=x -> x[1])
        m2 = candidates[1][2]
        m3 = candidates[1][3]
    else
        # Fallback to Bezout construction if search didn't close (rare)
        g_hk = gcd(h_c, k_c)
        if g_hk == 0
            m1 = [1, 0, 0]
        else
            m1 = [div(-k_c, g_hk), div(h_c, g_hk), 0]
        end
        m2_raw = cross([h_c, k_c, l_c], m1)
        g_m2 = gcd(gcd(m2_raw[1], m2_raw[2]), m2_raw[3])
        m2 = div.(m2_raw, g_m2)
        if g_hk == 0
            m3 = [0, 0, sign(l_c)]
        else
            g_all, s_hk, t_l = gcdx(g_hk, l_c)
            _, s_h, t_k = gcdx(h_c, k_c)
            m3 = [s_hk * s_h, s_hk * t_k, t_l]
        end
    end

    # Step 5: Enforce positive determinant det(M) > 0 for a right-handed cell coordinate system
    M = [m1 m2 m3]
    if round(Int, det(M)) < 0
        m1 = -m1
        M = [m1 m2 m3]
    end

    return M
end

"""
    rotation_matrix_to_z(u1::Vector{Float64}, u2::Vector{Float64}, u3::Vector{Float64}) -> Matrix{Float64}

Calculate an orthogonal 3x3 rotation matrix R in SO(3) that rotates the surface normal
u1 x u2 to align with (0, 0, 1) and places u1 in the Cartesian x-y plane.

# Mathematical Formulation
Let Q = [x_surf y_surf z_surf] be the orthonormal surface frame basis:
- x_surf = u1 / ||u1|| (unit vector along u1 in x-y plane)
- z_surf = (u1 x u2) / ||u1 x u2|| (unit surface normal)
- y_surf = z_surf x x_surf (completing right-handed basis)

The rotation matrix mapping crystal Cartesian coordinates to surface-aligned coordinates is:
    R = Q^T in SO(3)
"""
function rotation_matrix_to_z(u1::Vector{Float64}, u2::Vector{Float64}, u3::Vector{Float64})
    # Step 1: Unit vector along u1 defines Cartesian x-axis in surface plane
    x_surf = u1 / norm(u1)
    # Step 2: Unit surface normal vector n defines Cartesian z-axis
    n_surf = cross(u1, u2) / norm(cross(u1, u2))
    # Step 3: Unit vector y = z x x completes right-handed orthonormal basis Q
    y_surf = cross(n_surf, x_surf)

    Q = [x_surf y_surf n_surf]
    # Step 4: Transpose R = Q' yields orthogonal rotation matrix R in SO(3)
    R = Q'
    if det(R) < 0
        R = -R
    end
    return R
end

"""
    detect_molecules(crystal::Crystal) -> Vector{MoleculeData}

Extract connected components of `crystal.bonds` (running `infer_bonds!` if unassigned),
un-wrap periodic boundary crossings via Breadth-First Search (BFS), and extract intra-molecular
bonding edges for each molecule.

# Algorithm
1. **Graph Connected Components**: Connected sub-graphs of `crystal.bonds` define isolated molecules.
2. **BFS Periodic Un-wrapping**: For each molecule, traverse bonded neighbors from a root atom.
   For neighbor v of atom u, calculate nearest periodic image displacement:
       d_frac = c_to_f * (r_v - r_u) - round(c_to_f * (r_v - r_u))
       r_v_unwrap = r_u_unwrap + f_to_c * d_frac
3. **Centroid & Relative Coords**: Calculate Cartesian centroid c = sum(r_unwrap) / N_m and store relative positions.
"""
function detect_molecules(crystal::Crystal)
    c = deepcopy(crystal)
    # Step 1: Infer crystal bonding graph if unassigned
    if ne(c.bonds) == 0
        infer_bonds!(c, true)
    end

    # Step 2: Extract connected component atom index sets (molecules) from c.bonds
    conn_comps = connected_components(c.bonds)
    molecules = MoleculeData[]

    coords_cart = Cart(c.atoms.coords, c.box).x
    f_to_c = c.box.f_to_c
    c_to_f = c.box.c_to_f

    for comp in conn_comps
        n_m = length(comp)
        m_species = c.atoms.species[comp]
        m_charges = has_charges(c) ? c.charges.q[comp] : zeros(Float64, n_m)

        local_map = Dict{Int, Int}(comp[i] => i for i in 1:n_m)

        # Step 3: Breadth-First Search (BFS) coordinate un-wrapping along bonded neighbors
        unwrapped_cart = zeros(Float64, 3, n_m)
        visited = fill(false, n_m)

        queue = [comp[1]]
        visited[1] = true
        unwrapped_cart[:, 1] = coords_cart[:, comp[1]]

        while !isempty(queue)
            curr = popfirst!(queue)
            curr_local = local_map[curr]
            for nbr in neighbors(c.bonds, curr)
                if nbr in keys(local_map)
                    nbr_local = local_map[nbr]
                    if !visited[nbr_local]
                        visited[nbr_local] = true
                        diff_cart = coords_cart[:, nbr] - coords_cart[:, curr]
                        diff_frac = c_to_f * diff_cart
                        diff_frac .-= round.(diff_frac) # nearest image
                        unwrapped_cart[:, nbr_local] = unwrapped_cart[:, curr_local] + f_to_c * diff_frac
                        push!(queue, nbr)
                    end
                end
            end
        end

        # Step 4: Compute 3D Cartesian centroid c = (1/N_m) * sum(r_unwrap)
        centroid = vec(sum(unwrapped_cart, dims=2) ./ n_m)
        relative_cart = unwrapped_cart .- centroid

        # Step 5: Extract local intra-molecular bond edges (u, v) for slab graph reconstruction
        edges = Tuple{Int, Int}[]
        comp_set = Set(comp)
        for u in comp
            for v in neighbors(c.bonds, u)
                if v in comp_set && u < v
                    push!(edges, (local_map[u], local_map[v]))
                end
            end
        end

        push!(molecules, MoleculeData(comp, m_species, relative_cart, m_charges, centroid, edges))
    end

    return molecules
end

"""
    are_terminations_equivalent(crystal::Crystal, hkl::Tuple{Int, Int, Int},
                                z1::Float64, z2::Float64; tol::Float64=0.2) -> Bool

Determine if two surface cut shifts z1 and z2 produce structurally identical surface cuts
via direct 3D profile matching.

# Mathematical Formulation
For candidate cuts z1 and z2, project molecular centroids along rotated surface normal z:
    z_rel,m = mod((R * c_m)_3 - z_cut, h_layer) in [0, h_layer)

Construct sorted signature profiles:
    P(z_cut) = sort([(z_rel,m, species_sig_m)])

Return `true` if and only if P(z1) and P(z2) match in species sequence and relative heights
within distance tolerance `tol` (direct 3D translational equivalence).
"""
function are_terminations_equivalent(
    crystal::Crystal,
    hkl::Tuple{Int, Int, Int},
    z1::Float64,
    z2::Float64;
    tol::Float64=0.2
)
    geom = precompute_slab_geometry(crystal, hkl)
    h_layer = geom.u3_rot[3]

    # Step 1: Compute relative height profile and species signature for cut z1 and z2
    profiles1 = Tuple{Float64, Symbol}[]
    profiles2 = Tuple{Float64, Symbol}[]

    for mol in geom.molecules
        c_rot = geom.R * mol.centroid_cart
        sp_sig = Symbol(join(sort(string.(mol.species))))

        zh1 = mod(c_rot[3] - z1, h_layer)
        zh2 = mod(c_rot[3] - z2, h_layer)

        push!(profiles1, (zh1, sp_sig))
        push!(profiles2, (zh2, sp_sig))
    end

    sort!(profiles1, by=x -> (x[1], x[2]))
    sort!(profiles2, by=x -> (x[1], x[2]))

    # Step 2: Direct 3D translational matching: verify species and relative heights match within tol
    for i in 1:length(profiles1)
        if profiles1[i][2] != profiles2[i][2] || abs(profiles1[i][1] - profiles2[i][1]) > tol
            return false
        end
    end
    return true
end

"""
    get_terminations(crystal::Crystal, hkl::Tuple{Int, Int, Int};
                     tol::Float64=0.5, unique_only::Bool=false) -> Vector{Float64}

Identify distinct molecular surface termination plane heights along the (h,k,l) surface normal.
If `unique_only=true`, filter out structurally equivalent terminations using `are_terminations_equivalent`.
"""
function get_terminations(crystal::Crystal, hkl::Tuple{Int, Int, Int}; tol::Float64=0.5, unique_only::Bool=false)
    f_to_c_orig = crystal.box.f_to_c
    M = surface_basis(hkl; f_to_c=f_to_c_orig)
    u1 = f_to_c_orig * M[:, 1]
    u2 = f_to_c_orig * M[:, 2]
    u3 = f_to_c_orig * M[:, 3]

    R = rotation_matrix_to_z(u1, u2, u3)
    u3_rot = R * u3
    h_layer = u3_rot[3]

    molecules = detect_molecules(crystal)

    # Step 1: Project molecular centroids along z and wrap into single unit layer [0, h_layer)
    z_heights = Float64[]
    for mol in molecules
        c_rot = R * mol.centroid_cart
        z_mod = mod(c_rot[3], h_layer)
        push!(z_heights, z_mod)
    end
    sort!(z_heights)

    # Step 2: Agglomerative height clustering within distance tolerance `tol`
    clustered = Float64[]
    if !isempty(z_heights)
        current_cluster = [z_heights[1]]
        for i in 2:length(z_heights)
            if (z_heights[i] - current_cluster[end]) <= tol
                push!(current_cluster, z_heights[i])
            else
                push!(clustered, sum(current_cluster) / length(current_cluster))
                current_cluster = [z_heights[i]]
            end
        end
        push!(clustered, sum(current_cluster) / length(current_cluster))
    end

    if !unique_only || length(clustered) <= 1
        return clustered
    end

    # Step 3: Optional unique_only filtering using are_terminations_equivalent
    unique_terms = Float64[clustered[1]]
    for i in 2:length(clustered)
        is_equiv = false
        for u_term in unique_terms
            if are_terminations_equivalent(crystal, hkl, clustered[i], u_term; tol=tol)
                is_equiv = true
                break
            end
        end
        if !is_equiv
            push!(unique_terms, clustered[i])
        end
    end

    return unique_terms
end

"""
    check_slab_distances(slab::Crystal; min_dist::Float64=0.7) -> Float64

Calculate the minimum non-bonded atom-atom distance in the slab.
Throws a warning if any non-bonded atom pair is closer than `min_dist`.
"""
function check_slab_distances(slab::Crystal; min_dist::Float64=0.7)
    # Step 1: Extract Cartesian coordinates of slab atoms
    coords = Cart(slab.atoms.coords, slab.box).x
    n = slab.atoms.n
    min_d = 1e9

    # Step 2: Iterate over non-bonded atom pairs (skipping edges in slab.bonds)
    for i in 1:n, j in (i+1):n
        if has_edge(slab.bonds, i, j)
            continue
        end
        d = norm(coords[:, i] - coords[:, j])
        if d < min_d
            min_d = d
        end
    end

    # Step 3: Emit warning if minimum non-bonded distance < min_dist
    if min_d < min_dist
        @warn "Slab contains non-bonded atoms at distance $(round(min_d, digits=3)) Å < min_dist ($min_dist Å)"
    end

    return min_d
end

"""
    precompute_slab_geometry(crystal::Crystal, hkl::Tuple{Int, Int, Int})

Internal helper to pre-calculate transformation matrices, rotated cell vectors, and molecular data.
"""
function precompute_slab_geometry(crystal::Crystal, hkl::Tuple{Int, Int, Int})
    f_to_c_orig = crystal.box.f_to_c
    M = surface_basis(hkl; f_to_c=f_to_c_orig)
    u1 = f_to_c_orig * M[:, 1]
    u2 = f_to_c_orig * M[:, 2]
    u3 = f_to_c_orig * M[:, 3]

    R = rotation_matrix_to_z(u1, u2, u3)
    u1_rot = R * u1
    u2_rot = R * u2
    u3_rot = R * u3

    mols_base = detect_molecules(crystal)

    # Determine integer primitive cell shifts inside surface basis cell M
    det_M = abs(round(Int, det(M)))
    inv_M = inv(Float64.(M))
    primitive_shifts = Vector{Int}[]

    if det_M == 1
        push!(primitive_shifts, [0, 0, 0])
    else
        min_p = minimum(M, dims=2) .- 3
        max_p = maximum(M, dims=2) .+ 3
        for px in min_p[1]:max_p[1], py in min_p[2]:max_p[2], pz in min_p[3]:max_p[3]
            p = [px, py, pz]
            w = inv_M * p
            if -1e-7 <= w[1] < 1.0 - 1e-7 && -1e-7 <= w[2] < 1.0 - 1e-7 && -1e-7 <= w[3] < 1.0 - 1e-7
                push!(primitive_shifts, p)
            end
        end
    end

    # Expand base primitive molecules over all primitive cell shifts inside M
    molecules_expanded = MoleculeData[]
    for p in primitive_shifts
        p_cart = f_to_c_orig * p
        for m in mols_base
            c_shifted = m.centroid_cart + p_cart
            push!(molecules_expanded, MoleculeData(m.atom_indices, m.species, m.relative_coords_cart, m.charges, c_shifted, m.edges))
        end
    end

    return (M=M, R=R, u1_rot=u1_rot, u2_rot=u2_rot, u3_rot=u3_rot, molecules=molecules_expanded)
end

"""
    construct_single_slab(...) -> Crystal

Internal helper constructing an individual centered slab for `k_layer` with preserved molecular bonds.

# 3D Lattice Shift Math
When wrapping a molecule centroid across periodic layer boundaries:
    shift_z = floor((c_rot,3 - Z_term) / h_layer)

The full 3D Cartesian lattice vector shift -shift_z * u3_rot is applied across (x, y, z):
    c_base = c_rot - shift_z * u3_rot

For supercell grid indices rx, ry, kz:
    c_cell = c_base + rx * u1_rot + ry * u2_rot + kz * u3_rot
    c_centered = [c_cell,1, c_cell,2, (c_base,3 - Z_term) + kz * h_layer + vacuum]

This guarantees exact 3D monoclinic/triclinic lattice vector alignment with zero atom overlaps.
"""
function construct_single_slab(
    crystal::Crystal,
    hkl::Tuple{Int, Int, Int},
    geom,
    k_layer::Int;
    Nx::Int=1,
    Ny::Int=1,
    vacuum::Float64=10.0,
    termination::Int=1,
    shift::Float64=0.0,
    tol::Float64=0.5,
    orthogonalize_z::Bool=true,
    orthogonalize::Bool=false,
    keep_molecules_whole::Bool=true
)
    # Step 1: Determine total cell height including top and bottom vacuum padding
    h_layer = geom.u3_rot[3]
    Nz = k_layer
    slab_height_max = Nz * h_layer
    total_cell_z = slab_height_max + 2.0 * vacuum

    # Surface termination shift
    terminations = get_terminations(crystal, hkl; tol=tol)
    term_shift = 0.0
    if !isempty(terminations)
        idx = mod1(termination, length(terminations))
        term_shift = terminations[idx]
    end
    total_shift = term_shift + shift

    # Step 2: Construct supercell lattice vectors under requested orthogonalization mode
    if orthogonalize
        u1_slab = [norm(geom.u1_rot) * Nx, 0.0, 0.0]
        u2_slab = [0.0, norm(geom.u2_rot) * Ny, 0.0]
        u3_slab = [0.0, 0.0, total_cell_z]
    elseif orthogonalize_z
        u1_slab = geom.u1_rot * Nx
        u2_slab = geom.u2_rot * Ny
        u3_slab = [0.0, 0.0, total_cell_z]
    else
        u1_slab = geom.u1_rot * Nx
        u2_slab = geom.u2_rot * Ny
        u3_slab = [geom.u3_rot[1] * Nz, geom.u3_rot[2] * Nz, total_cell_z]
    end

    f_to_c_slab = [u1_slab u2_slab u3_slab]
    slab_box = Box(f_to_c_slab)

    # Step 3: Assemble atoms/molecules centered in cell with +vacuum z-offset
    slab_species = Symbol[]
    slab_coords_cart = Vector{Float64}[]
    slab_charges = Float64[]
    slab_edges = Tuple{Int, Int}[]

    if keep_molecules_whole && !isempty(geom.molecules)
        for mol in geom.molecules
            c_rot = geom.R * mol.centroid_cart
            z_rel = c_rot[3] - total_shift
            shift_z = floor(z_rel / h_layer)

            # Step 4: Calculate 3D lattice vector wrapping shift -shift_z * u3_rot for whole molecules
            c_base = c_rot - shift_z * geom.u3_rot
            z_mod = c_base[3] - total_shift

            for rx in 0:(Nx - 1), ry in 0:(Ny - 1), kz in 0:(Nz - 1)
                c_cell = c_base + rx * geom.u1_rot + ry * geom.u2_rot + kz * geom.u3_rot
                c_centered = [c_cell[1], c_cell[2], z_mod + kz * h_layer + vacuum]

                offset = length(slab_species)

                for i in 1:length(mol.species)
                    push!(slab_species, mol.species[i])
                    push!(slab_charges, mol.charges[i])
                    r_atom = c_centered + geom.R * mol.relative_coords_cart[:, i]
                    push!(slab_coords_cart, r_atom)
                end

                # Step 5: Re-map local molecular bond edges to global slab atom indices in slab_crystal.bonds
                for (lu, lv) in mol.edges
                    push!(slab_edges, (offset + lu, offset + lv))
                end
            end
        end
    else
        # Step 6: Fallback for atomic cut mode: apply 3D lattice shift -shift_z * u3_rot to individual atoms
        coords_cart_orig = Cart(crystal.atoms.coords, crystal.box).x
        for rx in 0:(Nx - 1), ry in 0:(Ny - 1), rz in 0:(Nz - 1)
            grid_shift = rx * geom.u1_rot + ry * geom.u2_rot
            for i in 1:crystal.atoms.n
                r_rot = geom.R * coords_cart_orig[:, i]
                z_rel = r_rot[3] - total_shift
                shift_z = floor(z_rel / h_layer)
                r_base = r_rot - shift_z * geom.u3_rot + grid_shift
                z_mod = r_base[3] - total_shift

                r_rot_centered = [r_base[1] + rz * geom.u3_rot[1],
                                  r_base[2] + rz * geom.u3_rot[2],
                                  z_mod + rz * h_layer + vacuum]
                push!(slab_species, crystal.atoms.species[i])
                if has_charges(crystal)
                    push!(slab_charges, crystal.charges.q[i])
                end
                push!(slab_coords_cart, r_rot_centered)
            end
        end
    end

    n_atoms_slab = length(slab_species)
    coords_mat = isempty(slab_coords_cart) ? zeros(Float64, 3, 0) : hcat(slab_coords_cart...)
    atoms_slab_cart = Cart(coords_mat)
    atoms_slab_frac = Frac(atoms_slab_cart, slab_box)
    atoms_slab = Atoms(slab_species, atoms_slab_frac)

    charges_slab = Charges(
        isempty(slab_charges) ? zeros(Float64, n_atoms_slab) : slab_charges,
        atoms_slab_frac
    )

    slab_name = "$(crystal.name)_slab_$(hkl[1])$(hkl[2])$(hkl[3])_L$(k_layer)"
    slab_crystal = Crystal(slab_name, slab_box, atoms_slab, charges_slab)

    # Populate bonds graph
    if keep_molecules_whole && !isempty(geom.molecules)
        for (u, v) in slab_edges
            add_edge!(slab_crystal.bonds, u, v)
        end
    elseif slab_crystal.atoms.n > 0
        infer_bonds!(slab_crystal, true)
    end

    return slab_crystal
end

"""
    slabs = make_slab(crystal::Crystal, hkl::Tuple{Int, Int, Int};
                      max_height::Union{Nothing, Float64}=nothing,
                      min_thickness::Union{Nothing, Float64}=nothing,
                      min_size::Union{Nothing, Tuple{Float64, Float64}}=nothing,
                      surface_dim::Tuple{Int, Int}=(1, 1),
                      layers::Union{Int, AbstractVector{Int}}=1,
                      vacuum::Float64=10.0,
                      termination::Int=1,
                      shift::Float64=0.0,
                      tol::Float64=0.5,
                      orthogonalize_z::Bool=true,
                      orthogonalize::Bool=false,
                      keep_molecules_whole::Bool=true) -> Vector{Crystal}

Construct surface slabs for Miller indices (h, k, l) returning a `Vector{Crystal}` containing
all generated intermediate layer slabs (from 1 layer up to N_max layers).

# Arguments & Keywords
- `crystal::Crystal`: Input crystal structure.
- `hkl::Tuple{Int, Int, Int}`: Miller indices (h, k, l) of the surface plane.
- `max_height::Union{Nothing, Float64}`: Target maximum slab height along z in Ångströms.
- `min_thickness::Union{Nothing, Float64}`: Target minimum slab thickness along z in Ångströms.
- `min_size::Union{Nothing, Tuple{Float64, Float64}}`: Minimum surface supercell dimensions (Lx, Ly) in Ångströms.
- `surface_dim::Tuple{Int, Int}`: Lateral supercell replication factors (Nx, Ny).
- `layers::Union{Int, AbstractVector{Int}}`: Target layer count or layer range.
- `vacuum::Float64`: Vacuum gap padding added to both top and bottom of the slab (Ångströms).
- `termination::Int`: Index of the molecular surface termination plane cut.
- `orthogonalize_z::Bool`: Force alpha = 90 deg and beta = 90 deg cell angles (default: `true`).
- `orthogonalize::Bool`: Force alpha = 90 deg, beta = 90 deg, and gamma = 90 deg cell angles.
- `keep_molecules_whole::Bool`: Preserve intact molecules across surface cuts (default: `true`).
"""
function make_slab(
    crystal::Crystal,
    hkl::Tuple{Int, Int, Int};
    max_height::Union{Nothing, Float64}=nothing,
    min_thickness::Union{Nothing, Float64}=nothing,
    min_size::Union{Nothing, Tuple{Float64, Float64}}=nothing,
    surface_dim::Tuple{Int, Int}=(1, 1),
    layers::Union{Int, AbstractVector{Int}}=1,
    vacuum::Float64=10.0,
    termination::Int=1,
    shift::Float64=0.0,
    tol::Float64=0.5,
    orthogonalize_z::Bool=true,
    orthogonalize::Bool=false,
    keep_molecules_whole::Bool=true
)
    # Step 1: Input validation guard clauses for vacuum, dimensions, and layer counts
    if vacuum < 0.0
        throw(ArgumentError("vacuum padding must be non-negative, got $vacuum"))
    end
    if max_height !== nothing && max_height <= 0.0
        throw(ArgumentError("max_height must be positive, got $max_height"))
    end
    if min_thickness !== nothing && min_thickness <= 0.0
        throw(ArgumentError("min_thickness must be positive, got $min_thickness"))
    end
    if min_size !== nothing && (min_size[1] <= 0.0 || min_size[2] <= 0.0)
        throw(ArgumentError("min_size dimensions must be positive, got $min_size"))
    end
    if layers isa Int && layers < 1
        throw(ArgumentError("layers must be >= 1, got $layers"))
    end

    geom = precompute_slab_geometry(crystal, hkl)

    # Step 2: Calculate minimum lateral surface replication factors (Nx, Ny)
    L1 = norm(geom.u1_rot)
    L2 = norm(geom.u2_rot)

    Nx, Ny = surface_dim
    if min_size !== nothing
        Nx = max(1, ceil(Int, min_size[1] / L1))
        Ny = max(1, ceil(Int, min_size[2] / L2))
    end

    # Step 3: Determine maximum layer count N_max
    h_layer = geom.u3_rot[3]
    N_max = 1

    if max_height !== nothing
        N_max = max(1, ceil(Int, max_height / h_layer))
    elseif min_thickness !== nothing
        N_max = max(1, ceil(Int, min_thickness / h_layer))
    elseif layers isa AbstractVector
        N_max = maximum(layers)
    else
        N_max = layers
    end

    slabs = Vector{Crystal}(undef, N_max)

    # Step 4: Parallel execution over layer series k in 1:N_max via Threads.@threads
    Threads.@threads for k in 1:N_max
        slabs[k] = construct_single_slab(
            crystal,
            hkl,
            geom,
            k;
            Nx=Nx,
            Ny=Ny,
            vacuum=vacuum,
            termination=termination,
            shift=shift,
            tol=tol,
            orthogonalize_z=orthogonalize_z,
            orthogonalize=orthogonalize,
            keep_molecules_whole=keep_molecules_whole
        )
    end

    return slabs
end

# Alias make_slabs -> make_slab
const make_slabs = make_slab
