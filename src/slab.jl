using LinearAlgebra
using Graphs
using StaticArrays

"""
    MoleculeData(atom_indices, species, relative_coords_cart, charges, centroid_cart, edges)

Data container holding pre-computed information and bonding edges for an individual molecule.
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
    surface_basis(hkl::Tuple{Int, Int, Int}) -> Matrix{Int}

Compute a 3x3 integer transformation matrix M = [m1 m2 m3] for Miller indices (h, k, l)
where m1 and m2 lie in the (h, k, l) plane, m3 is out-of-plane, and det(M) > 0.
"""
function surface_basis(hkl::Tuple{Int, Int, Int})
    h, k, l = hkl
    if h == 0 && k == 0 && l == 0
        throw(ArgumentError("Miller indices cannot be (0, 0, 0)"))
    end

    g_hkl = gcd(gcd(h, k), l)
    h_c, k_c, l_c = div(h, g_hkl), div(k, g_hkl), div(l, g_hkl)
    v_hkl = [h_c, k_c, l_c]

    # In-plane vector m1
    g_hk = gcd(h_c, k_c)
    if g_hk == 0
        m1 = [1, 0, 0]
    else
        m1 = [div(-k_c, g_hk), div(h_c, g_hk), 0]
    end

    # In-plane vector m2 perpendicular to v_hkl and m1
    m2_raw = cross(v_hkl, m1)
    g_m2 = gcd(gcd(m2_raw[1], m2_raw[2]), m2_raw[3])
    m2 = div.(m2_raw, g_m2)

    # Out-of-plane vector m3 via Bezout
    if g_hk == 0
        m3 = [0, 0, sign(l_c)]
    else
        g_all, s_hk, t_l = gcdx(g_hk, l_c)
        _, s_h, t_k = gcdx(h_c, k_c)
        m3 = [s_hk * s_h, s_hk * t_k, t_l]
    end

    det_M = round(Int, det([m1 m2 m3]))
    if det_M < 0
        m1 = -m1
        det_M = round(Int, det([m1 m2 m3]))
    end

    M = [m1 m2 m3]
    return M
end

"""
    rotation_matrix_to_z(u1::Vector{Float64}, u2::Vector{Float64}, u3::Vector{Float64}) -> Matrix{Float64}

Calculate an orthogonal 3x3 rotation matrix R in SO(3) that rotates the surface normal
u1 x u2 to align with (0, 0, 1) and places u1 in the Cartesian x-y plane.
"""
function rotation_matrix_to_z(u1::Vector{Float64}, u2::Vector{Float64}, u3::Vector{Float64})
    x_surf = u1 / norm(u1)
    n_surf = cross(u1, u2) / norm(cross(u1, u2))
    y_surf = cross(n_surf, x_surf)

    Q = [x_surf y_surf n_surf]
    R = Q'
    if det(R) < 0
        R = -R
    end
    return R
end

"""
    detect_molecules(crystal::Crystal) -> Vector{MoleculeData}

Extract connected components of crystal.bonds (running infer_bonds! if unassigned),
un-wrap periodic boundary crossings via BFS, and extract intra-molecular bonding edges for each molecule.
"""
function detect_molecules(crystal::Crystal)
    c = deepcopy(crystal)
    if ne(c.bonds) == 0
        infer_bonds!(c, true)
    end

    conn_comps = connected_components(c.bonds)
    molecules = MoleculeData[]

    # Cartesian coordinates
    coords_cart = Cart(c.atoms.coords, c.box).x
    f_to_c = c.box.f_to_c
    c_to_f = c.box.c_to_f

    for comp in conn_comps
        n_m = length(comp)
        m_species = c.atoms.species[comp]
        m_charges = has_charges(c) ? c.charges.q[comp] : zeros(Float64, n_m)

        # Local atom index map
        local_map = Dict{Int, Int}(comp[i] => i for i in 1:n_m)

        # BFS un-wrap relative to first atom
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

        # Calculate Cartesian centroid
        centroid = vec(sum(unwrapped_cart, dims=2) ./ n_m)
        relative_cart = unwrapped_cart .- centroid

        # Extract intra-molecular edges from c.bonds
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
    get_terminations(crystal::Crystal, hkl::Tuple{Int, Int, Int}; tol::Float64=0.5) -> Vector{Float64}

Identify distinct molecular surface termination plane heights along the (h,k,l) surface normal.
"""
function get_terminations(crystal::Crystal, hkl::Tuple{Int, Int, Int}; tol::Float64=0.5)
    M = surface_basis(hkl)
    f_to_c_orig = crystal.box.f_to_c
    u1 = f_to_c_orig * M[:, 1]
    u2 = f_to_c_orig * M[:, 2]
    u3 = f_to_c_orig * M[:, 3]

    R = rotation_matrix_to_z(u1, u2, u3)
    u3_rot = R * u3
    h_layer = u3_rot[3]

    molecules = detect_molecules(crystal)

    # Project centroids along z and wrap into single unit layer [0, h_layer)
    z_heights = Float64[]
    for mol in molecules
        c_rot = R * mol.centroid_cart
        z_mod = mod(c_rot[3], h_layer)
        push!(z_heights, z_mod)
    end
    sort!(z_heights)

    # Cluster heights within distance tolerance `tol`
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

    return clustered
end

"""
    precompute_slab_geometry(crystal::Crystal, hkl::Tuple{Int, Int, Int})

Internal helper to pre-calculate transformation matrices, rotated cell vectors, and molecular data.
"""
function precompute_slab_geometry(crystal::Crystal, hkl::Tuple{Int, Int, Int})
    M = surface_basis(hkl)
    f_to_c_orig = crystal.box.f_to_c
    u1 = f_to_c_orig * M[:, 1]
    u2 = f_to_c_orig * M[:, 2]
    u3 = f_to_c_orig * M[:, 3]

    R = rotation_matrix_to_z(u1, u2, u3)
    u1_rot = R * u1
    u2_rot = R * u2
    u3_rot = R * u3

    molecules = detect_molecules(crystal)

    return (M=M, R=R, u1_rot=u1_rot, u2_rot=u2_rot, u3_rot=u3_rot, molecules=molecules)
end

"""
    construct_single_slab(...) -> Crystal

Internal helper constructing an individual centered slab for k_layer with preserved molecular bonds.
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

    # Cell vectors
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

    # Assemble atoms/molecules centered with vacuum at top & bottom (+ vacuum offset in z)
    slab_species = Symbol[]
    slab_coords_cart = Vector{Float64}[]
    slab_charges = Float64[]
    slab_edges = Tuple{Int, Int}[]

    if keep_molecules_whole && !isempty(geom.molecules)
        for mol in geom.molecules
            c_rot = geom.R * mol.centroid_cart
            z_rel = c_rot[3] - total_shift
            shift_z = floor(z_rel / h_layer)
            z_mod = z_rel - shift_z * h_layer
            c_base = [c_rot[1], c_rot[2], c_rot[3] - shift_z * h_layer]

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

                # Preserve molecular bond edges in slab
                for (lu, lv) in mol.edges
                    push!(slab_edges, (offset + lu, offset + lv))
                end
            end
        end
    else
        coords_cart_orig = Cart(crystal.atoms.coords, crystal.box).x
        for rx in 0:(Nx - 1), ry in 0:(Ny - 1), rz in 0:(Nz - 1)
            grid_shift = rx * geom.u1_rot + ry * geom.u2_rot + rz * geom.u3_rot
            for i in 1:crystal.atoms.n
                r_rot = geom.R * coords_cart_orig[:, i] + grid_shift
                r_z_rel = r_rot[3] - total_shift
                if 0.0 <= r_z_rel < slab_height_max
                    r_rot_centered = [r_rot[1], r_rot[2], r_rot[3] - total_shift + vacuum]
                    push!(slab_species, crystal.atoms.species[i])
                    if has_charges(crystal)
                        push!(slab_charges, crystal.charges.q[i])
                    end
                    push!(slab_coords_cart, r_rot_centered)
                end
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

Construct surface slabs for Miller indices (h, k, l) returning a Vector{Crystal} containing
all generated intermediate layer slabs (from 1 layer up to N_max layers).
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
    # Guard clauses and input validation
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

    # Determine lateral surface replication Nx, Ny
    L1 = norm(geom.u1_rot)
    L2 = norm(geom.u2_rot)

    Nx, Ny = surface_dim
    if min_size !== nothing
        Nx = max(1, ceil(Int, min_size[1] / L1))
        Ny = max(1, ceil(Int, min_size[2] / L2))
    end

    # Determine layer range 1:N_max
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

    # Parallel construction over all layer counts
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
