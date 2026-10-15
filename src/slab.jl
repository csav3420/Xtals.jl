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
function surface_basis(
    hkl::Tuple{Int, Int, Int}; f_to_c::Union{Nothing, Matrix{Float64}}=nothing
)
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
        sort!(in_plane; by=m -> norm(f_to_c * m))
    else
        sort!(in_plane; by=m -> norm(m))
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
        sort!(candidates; by=x -> x[1])
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
                        unwrapped_cart[:, nbr_local] =
                            unwrapped_cart[:, curr_local] + f_to_c * diff_frac
                        push!(queue, nbr)
                    end
                end
            end
        end

        # Step 4: Compute 3D Cartesian centroid c = (1/N_m) * sum(r_unwrap)
        centroid = vec(sum(unwrapped_cart; dims=2) ./ n_m)
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

        push!(
            molecules,
            MoleculeData(comp, m_species, relative_cart, m_charges, centroid, edges),
        )
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
# Private helper to parse top and bottom termination indices from single/tuple/keyword options
function _parse_terminations(
    termination::Union{Int, Tuple{Int, Int}},
    termination_bottom::Union{Nothing, Int},
    termination_top::Union{Nothing, Int},
)
    t_bot = if termination_bottom !== nothing
        termination_bottom
    else
        (termination isa Tuple ? termination[1] : termination)
    end
    t_top = if termination_top !== nothing
        termination_top
    else
        (termination isa Tuple ? termination[2] : termination)
    end
    return t_bot, t_top
end

function are_terminations_equivalent(geom, z1::Float64, z2::Float64; tol::Float64=0.2)
    h_layer = geom.u3_rot[3]
    n_mols = length(geom.molecules)

    profiles1 = Vector{Tuple{Float64, Symbol}}()
    profiles2 = Vector{Tuple{Float64, Symbol}}()
    sizehint!(profiles1, n_mols)
    sizehint!(profiles2, n_mols)

    if !isempty(geom.molecules)
        for mol in geom.molecules
            c_rot_z = (geom.R * mol.centroid_cart)[3]
            sp_sig = Symbol(join(sort(string.(mol.species))))

            push!(profiles1, (mod(c_rot_z - z1, h_layer), sp_sig))
            push!(profiles2, (mod(c_rot_z - z2, h_layer), sp_sig))
        end
    end

    sort!(profiles1; by=x -> (x[1], x[2]))
    sort!(profiles2; by=x -> (x[1], x[2]))

    for i in 1:length(profiles1)
        if profiles1[i][2] != profiles2[i][2] ||
            abs(profiles1[i][1] - profiles2[i][1]) > tol
            return false
        end
    end
    return true
end

function are_terminations_equivalent(
    crystal::Crystal, hkl::Tuple{Int, Int, Int}, z1::Float64, z2::Float64; tol::Float64=0.2
)
    geom = precompute_slab_geometry(crystal, hkl)
    return are_terminations_equivalent(geom, z1, z2; tol=tol)
end

"""
    get_terminations(crystal::Crystal, hkl::Tuple{Int, Int, Int};
                     tol::Float64=0.5, unique_only::Bool=false, nonpolar_only::Bool=false) -> Vector{Float64}

Identify distinct molecular surface termination plane heights along the (h,k,l) surface normal.
- If `unique_only=true`, filter out structurally equivalent terminations using `are_terminations_equivalent`.
- If `nonpolar_only=true`, filter out polar surface terminations (|mu_z| > 1e-3).
"""
function get_terminations(
    crystal::Crystal,
    hkl::Tuple{Int, Int, Int};
    tol::Float64=0.5,
    unique_only::Bool=false,
    nonpolar_only::Bool=false,
)
    geom = precompute_slab_geometry(crystal, hkl)
    R = geom.R
    h_layer = geom.u3_rot[3]

    # Step 1: Project molecular centroids (or atomic positions) along z into single unit layer [0, h_layer)
    z_heights = Float64[]
    if !isempty(geom.molecules)
        for mol in geom.molecules
            c_rot = R * mol.centroid_cart
            push!(z_heights, mod(c_rot[3], h_layer))
        end
    else
        f_to_c_orig = crystal.box.f_to_c
        for i in 1:crystal.atoms.n
            r_rot = R * (f_to_c_orig * crystal.atoms.coords.xf[:, i])
            push!(z_heights, mod(r_rot[3], h_layer))
        end
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

    filtered_terms = clustered
    if unique_only && length(clustered) > 1
        unique_terms = Float64[clustered[1]]
        for i in 2:length(clustered)
            is_equiv = false
            for u_term in unique_terms
                if are_terminations_equivalent(geom, clustered[i], u_term; tol=tol)
                    is_equiv = true
                    break
                end
            end
            if !is_equiv
                push!(unique_terms, clustered[i])
            end
        end
        filtered_terms = unique_terms
    end

    # Step 4: Optional nonpolar_only filtering
    if nonpolar_only && has_charges(crystal)
        nonpolar_terms = Float64[]
        for t_val in filtered_terms
            test_slab = construct_single_slab(
                crystal, hkl, geom, 1; termination=1, shift=t_val
            )
            if !is_polar_surface(test_slab)
                push!(nonpolar_terms, t_val)
            end
        end
        return nonpolar_terms
    end

    return filtered_terms
end

"""
    check_slab_distances(slab::Crystal; min_dist::Float64=0.7) -> Float64

Calculate the minimum non-bonded atom-atom distance in the slab.
"""
function check_slab_distances(slab::Crystal; min_dist::Float64=0.7)
    c = Cart(slab.atoms.coords, slab.box).x
    n = slab.atoms.n
    min_d = Inf
    for i in 1:(n - 1), j in (i + 1):n
        if has_edge(slab.bonds, i, j)
            continue
        end
        d = norm(c[:, i] - c[:, j])
        if d < min_d
            min_d = d
        end
    end
    return min_d
end

"""
    find_near_square_supercell(u1::Vector{Float64}, u2::Vector{Float64};
                               min_size::Union{Nothing, Tuple{Float64, Float64}, Float64}=nothing,
                               max_det::Int=24) -> NamedTuple

Find a 2D integer transformation matrix N = [n11 n12; n21 n22] in Z^(2x2) with det(N) > 0
that transforms primitive in-plane surface vectors [u1 u2] into a near-square supercell
(gamma ≈ 90°, ||w1|| ≈ ||w2||) with zero artificial coordinate strain.

# Arguments
- `u1::Vector{Float64}`: First in-plane primitive surface basis vector (in surface Cartesian frame).
- `u2::Vector{Float64}`: Second in-plane primitive surface basis vector.
- `min_size`: Optional minimum target lateral dimension (Lx, Ly).
- `max_det::Int`: Maximum supercell area multiplication factor (default: 24).

# Returns
- `NamedTuple`: (N, det, w1, w2, L1, L2, gamma_deg)
"""
function find_near_square_supercell(
    u1::Vector{Float64},
    u2::Vector{Float64};
    min_size::Union{Nothing, Tuple{Float64, Float64}, Float64}=nothing,
    max_det::Int=24,
)
    min_L = if min_size === nothing
        5.0
    elseif min_size isa Tuple{Float64, Float64}
        max(min_size[1], min_size[2])
    else
        Float64(min_size)
    end

    area_prim = abs(u1[1] * u2[2] - u1[2] * u2[1])
    if min_size !== nothing && area_prim > 1e-4
        min_det_req = ceil(Int, (min_L * min_L) / area_prim)
        max_det = max(max_det, min_det_req + 12)
    end

    best_score = Inf
    u1_len = norm(u1)
    u2_len = norm(u2)
    best_result = (
        N=[1 0; 0 1],
        det=1,
        w1=u1,
        w2=u2,
        L1=u1_len,
        L2=u2_len,
        gamma_deg=acos(clamp(dot(u1, u2) / (u1_len * u2_len), -1.0, 1.0)) * 180.0 / π,
    )

    max_idx = max(8, ceil(Int, sqrt(max_det) * 2))

    for n11 in (-max_idx):max_idx, n21 in (-max_idx):max_idx
        w1 = n11 * u1 + n21 * u2
        L1 = norm(w1)
        if L1 < min_L
            continue
        end

        for n12 in (-max_idx):max_idx, n22 in (-max_idx):max_idx
            det2d = n11 * n22 - n12 * n21
            if det2d <= 0 || det2d > max_det
                continue
            end

            w2 = n12 * u1 + n22 * u2
            L2 = norm(w2)
            if L2 < min_L
                continue
            end

            cos_ang = dot(w1, w2) / (L1 * L2)
            ang_deg = acos(clamp(cos_ang, -1.0, 1.0)) * 180.0 / π
            ang_dev = abs(ang_deg - 90.0)
            len_ratio = max(L1 / L2, L2 / L1)

            # Optimization score: prioritize perpendicular angle, square aspect ratio, and compact area
            score = ang_dev + 35.0 * (len_ratio - 1.0) + 0.15 * det2d

            if score < best_score
                best_score = score
                best_result = (
                    N=[n11 n12; n21 n22],
                    det=det2d,
                    w1=w1,
                    w2=w2,
                    L1=L1,
                    L2=L2,
                    gamma_deg=ang_deg,
                )
            end
        end
    end

    return best_result
end

"""
    enumerate_supercell_grid_points(N::Matrix{Int}) -> Vector{Tuple{Int, Int}}

Enumerate all exact integer lattice translation points (r1, r2) in Z^2 that map inside
the 2D supercell spanned by integer matrix N (where inv(N) * [r1, r2] in [0, 1)^2).
Guarantees exactly det(N) non-overlapping grid points.
"""
function enumerate_supercell_grid_points(N::Matrix{Int})
    detN = round(Int, det(N))
    invN = inv(N)

    r1_min = min(0, N[1, 1], N[1, 2], N[1, 1] + N[1, 2]) - 2
    r1_max = max(0, N[1, 1], N[1, 2], N[1, 1] + N[1, 2]) + 2
    r2_min = min(0, N[2, 1], N[2, 2], N[2, 1] + N[2, 2]) - 2
    r2_max = max(0, N[2, 1], N[2, 2], N[2, 1] + N[2, 2]) + 2

    grid_points = Tuple{Int, Int}[]
    sizehint!(grid_points, detN)

    for r1 in r1_min:r1_max, r2 in r2_min:r2_max
        s = invN * [r1, r2]
        if s[1] >= -1e-6 && s[1] < 1.0 - 1e-6 && s[2] >= -1e-6 && s[2] < 1.0 - 1e-6
            push!(grid_points, (r1, r2))
        end
    end

    return grid_points
end

"""
    precompute_slab_geometry(crystal::Crystal, hkl::Tuple{Int, Int, Int}) -> NamedTuple

Precomputes primitive surface basis matrix `M`, lattice rotation matrix `R`,
rotated surface cell vectors `u1_rot`, `u2_rot`, `u3_rot`, and whole molecular clusters.
"""
function precompute_slab_geometry(crystal::Crystal, hkl::Tuple{Int, Int, Int})
    f_to_c = crystal.box.f_to_c
    M = surface_basis(hkl; f_to_c=f_to_c)
    u1 = f_to_c * M[:, 1]
    u2 = f_to_c * M[:, 2]
    u3 = f_to_c * M[:, 3]

    R = rotation_matrix_to_z(u1, u2, u3)
    u1_rot = R * u1
    u2_rot = R * u2
    u3_rot = R * u3

    molecules = detect_molecules(crystal)

    return (M=M, R=R, u1_rot=u1_rot, u2_rot=u2_rot, u3_rot=u3_rot, molecules=molecules)
end

"""
    construct_single_slab(crystal, hkl, geom, k_layer; kwargs...) -> Crystal

Constructs a single 3D surface slab for a given layer count `k_layer` under `geom`.

This guarantees exact 3D monoclinic/triclinic lattice vector alignment with zero atom overlaps.
"""
function construct_single_slab(
    crystal::Crystal,
    hkl::Tuple{Int, Int, Int},
    geom,
    k_layer::Int;
    Nx::Int=1,
    Ny::Int=1,
    supercell_N::Union{Nothing, Matrix{Int}}=nothing,
    grid_points::Union{Nothing, Vector{Tuple{Int, Int}}}=nothing,
    vacuum::Float64=10.0,
    termination::Union{Int, Tuple{Int, Int}}=1,
    termination_bottom::Union{Nothing, Int}=nothing,
    termination_top::Union{Nothing, Int}=nothing,
    shift::Float64=0.0,
    term_shift::Union{Nothing, Float64, Tuple{Float64, Float64}}=nothing,
    tol::Float64=0.5,
    orthogonalize_z::Bool=true,
    orthogonalize::Bool=false,
    keep_molecules_whole::Bool=true,
)
    # Parse bottom and top termination indices
    t_bot, t_top = _parse_terminations(termination, termination_bottom, termination_top)

    # Step 1: Determine total cell height including top and bottom vacuum padding
    h_layer = geom.u3_rot[3]
    Nz = k_layer

    s_bot = 0.0
    s_top = 0.0
    if term_shift isa Tuple{Float64, Float64}
        s_bot, s_top = term_shift
    elseif term_shift isa Float64
        s_bot = term_shift
        s_top = term_shift
    else
        terminations = get_terminations(crystal, hkl; tol=tol)
        if !isempty(terminations)
            idx_bot = mod1(t_bot, length(terminations))
            idx_top = mod1(t_top, length(terminations))
            s_bot = terminations[idx_bot]
            s_top = terminations[idx_top]
        end
    end

    shift_bot = s_bot + shift
    shift_top = s_top + shift

    delta_shift = shift_top - shift_bot
    total_slab_z = Nz * h_layer + delta_shift
    total_cell_z = total_slab_z + 2.0 * vacuum

    # Step 2: Construct supercell lattice vectors under requested orthogonalization mode
    if supercell_N !== nothing
        u1_slab = supercell_N[1, 1] * geom.u1_rot + supercell_N[2, 1] * geom.u2_rot
        u2_slab = supercell_N[1, 2] * geom.u1_rot + supercell_N[2, 2] * geom.u2_rot
        u3_slab = [0.0, 0.0, total_cell_z]
    elseif orthogonalize
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

    # Resolve in-plane grid points
    grid_pts = if grid_points !== nothing
        grid_points
    else
        [(rx, ry) for rx in 0:(Nx - 1) for ry in 0:(Ny - 1)]
    end
    n_in_plane = length(grid_pts)

    # Memory pre-allocation (sizehint!)
    slab_species = Symbol[]
    slab_coords_cart = Vector{Float64}[]
    slab_charges = Float64[]
    slab_edges = Tuple{Int, Int}[]

    if keep_molecules_whole && !isempty(geom.molecules)
        total_atoms_est = sum(length(m.species) for m in geom.molecules) * n_in_plane * Nz
        sizehint!(slab_species, total_atoms_est)
        sizehint!(slab_coords_cart, total_atoms_est)
        sizehint!(slab_charges, total_atoms_est)

        for mol in geom.molecules
            c_rot = geom.R * mol.centroid_cart
            z_rel = c_rot[3] - shift_bot
            shift_z = floor(z_rel / h_layer)

            # Step 4: Calculate 3D lattice vector wrapping shift -shift_z * u3_rot for whole molecules
            c_base = c_rot - shift_z * geom.u3_rot
            z_mod = c_base[3] - shift_bot

            max_kz = delta_shift > 1e-5 ? Nz : (Nz - 1)
            for (rx, ry) in grid_pts, kz in 0:max_kz
                if kz == Nz && z_mod > delta_shift + 1e-5
                    continue
                end

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
        total_atoms_est = crystal.atoms.n * n_in_plane * Nz
        sizehint!(slab_species, total_atoms_est)
        sizehint!(slab_coords_cart, total_atoms_est)
        if has_charges(crystal)
            sizehint!(slab_charges, total_atoms_est)
        end

        # Step 6: Fallback for atomic cut mode: apply 3D lattice shift -shift_z * u3_rot to individual atoms
        coords_cart_orig = Cart(crystal.atoms.coords, crystal.box).x
        r_rot_base = [geom.R * coords_cart_orig[:, i] for i in 1:crystal.atoms.n]

        max_rz = delta_shift > 1e-5 ? Nz : (Nz - 1)
        for (rx, ry) in grid_pts, rz in 0:max_rz
            grid_shift = rx * geom.u1_rot + ry * geom.u2_rot
            for i in 1:crystal.atoms.n
                r_rot = r_rot_base[i]
                z_rel = r_rot[3] - shift_bot
                shift_z = floor(z_rel / h_layer)
                r_base = r_rot - shift_z * geom.u3_rot + grid_shift
                z_mod = r_base[3] - shift_bot

                if rz == Nz && z_mod > delta_shift + 1e-5
                    continue
                end

                r_rot_centered = [
                    r_base[1] + rz * geom.u3_rot[1],
                    r_base[2] + rz * geom.u3_rot[2],
                    z_mod + rz * h_layer + vacuum,
                ]
                push!(slab_species, crystal.atoms.species[i])
                if has_charges(crystal)
                    push!(slab_charges, crystal.charges.q[i])
                end
                push!(slab_coords_cart, r_rot_centered)
            end
        end
    end

    n_atoms_slab = length(slab_species)
    coords_mat =
        isempty(slab_coords_cart) ? zeros(Float64, 3, 0) : hcat(slab_coords_cart...)
    atoms_slab_cart = Cart(coords_mat)
    atoms_slab_frac = Frac(atoms_slab_cart, slab_box)
    atoms_slab = Atoms(slab_species, atoms_slab_frac)

    charges_slab = Charges(
        isempty(slab_charges) ? zeros(Float64, n_atoms_slab) : slab_charges, atoms_slab_frac
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
                      termination::Union{Int, Tuple{Int, Int}}=1,
                      termination_bottom::Union{Nothing, Int}=nothing,
                      termination_top::Union{Nothing, Int}=nothing,
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
- `termination::Union{Int, Tuple{Int, Int}}`: Index or (bottom, top) tuple of surface termination cuts.
- `termination_bottom::Union{Nothing, Int}`: Explicit bottom surface termination cut index.
- `termination_top::Union{Nothing, Int}`: Explicit top surface termination cut index.
- `orthogonalize_z::Bool`: Force alpha = 90 deg and beta = 90 deg cell angles (default: `true`).
- `orthogonalize::Bool`: Force alpha = 90 deg, beta = 90 deg, and gamma = 90 deg cell angles.
- `keep_molecules_whole::Bool`: Preserve intact molecules across surface cuts (default: `true`).
- `near_square::Bool`: Construct a pure-integer near-square surface supercell (gamma ≈ 90°, ||w1|| ≈ ||w2||) with zero artificial strain (default: `false`).
- `max_supercell_det::Int`: Maximum supercell area multiplication factor for near-square search (default: `24`).
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
    termination::Union{Int, Tuple{Int, Int}}=1,
    termination_bottom::Union{Nothing, Int}=nothing,
    termination_top::Union{Nothing, Int}=nothing,
    shift::Float64=0.0,
    tol::Float64=0.5,
    orthogonalize_z::Bool=true,
    orthogonalize::Bool=false,
    keep_molecules_whole::Bool=true,
    near_square::Bool=false,
    max_supercell_det::Int=24,
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

    # Parse bottom and top termination indices
    t_bot, t_top = _parse_terminations(termination, termination_bottom, termination_top)

    # Calculate termination shifts ONCE at top-level
    terminations = get_terminations(crystal, hkl; tol=tol)
    term_shift_bot = 0.0
    term_shift_top = 0.0
    if !isempty(terminations)
        idx_bot = mod1(t_bot, length(terminations))
        idx_top = mod1(t_top, length(terminations))
        term_shift_bot = terminations[idx_bot]
        term_shift_top = terminations[idx_top]
    end
    term_shift_tuple = (term_shift_bot, term_shift_top)

    # Step 2: Calculate lateral surface supercell
    L1 = norm(geom.u1_rot)
    L2 = norm(geom.u2_rot)

    supercell_N = nothing
    grid_points = nothing
    Nx, Ny = surface_dim

    if near_square
        near_sq = find_near_square_supercell(
            geom.u1_rot, geom.u2_rot; min_size=min_size, max_det=max_supercell_det
        )
        supercell_N = near_sq.N
        grid_points = enumerate_supercell_grid_points(supercell_N)
    else
        if min_size !== nothing
            Nx = max(1, ceil(Int, min_size[1] / L1))
            Ny = max(1, ceil(Int, min_size[2] / L2))
        end
        grid_points = [(rx, ry) for rx in 0:(Nx - 1) for ry in 0:(Ny - 1)]
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
            supercell_N=supercell_N,
            grid_points=grid_points,
            vacuum=vacuum,
            termination=termination,
            termination_bottom=termination_bottom,
            termination_top=termination_top,
            shift=shift,
            term_shift=term_shift_tuple,
            tol=tol,
            orthogonalize_z=orthogonalize_z,
            orthogonalize=orthogonalize,
            keep_molecules_whole=keep_molecules_whole,
        )
    end

    if layers isa AbstractVector
        return slabs[layers]
    end

    return slabs
end

# Alias make_slabs -> make_slab
const make_slabs = make_slab

"""
    dipole_moment(crystal::Crystal; debye::Bool=false) -> Vector{Float64}

Calculate the 3D electric dipole moment vector mu = sum(q_i * r_i) of a crystal or slab
in units of e*Å (or Debye if debye=true, where 1 e*Å = 4.803204 Debye).

# Mathematical Formulation
    mu = sum_{i=1}^N q_i * r_i  in  R^3
"""
function dipole_moment(crystal::Crystal; debye::Bool=false)
    if !has_charges(crystal) || iszero(crystal.charges.q)
        throw(
            ArgumentError(
                "Crystal has no atomic charges assigned. Assign charges before calculating dipole moment.",
            ),
        )
    end
    coords = Cart(crystal.atoms.coords, crystal.box).x
    q = crystal.charges.q
    mu = vec(sum(q[i] .* coords[:, i] for i in 1:crystal.atoms.n))
    if debye
        mu .*= 4.803204
    end
    return mu
end

"""
    surface_dipole_moment(slab::Crystal; debye::Bool=false) -> Float64

Extract the out-of-plane surface normal electric dipole moment mu_z = sum(q_i * z_i) of a slab
in e*Å (or Debye if debye=true).
"""
function surface_dipole_moment(slab::Crystal; debye::Bool=false)
    mu = dipole_moment(slab; debye=debye)
    return mu[3]
end

"""
    is_polar_surface(slab::Crystal; tol::Float64=1e-3) -> Bool

Determine if a surface slab is polar (|mu_z| > tol e*Å) or non-polar (|mu_z| <= tol e*Å).
Polar surfaces with non-zero out-of-plane electric dipole moments are energetically unstable in nature.
"""
function is_polar_surface(slab::Crystal; tol::Float64=1e-3)
    return abs(surface_dipole_moment(slab)) > tol
end

"""
    layer_dipoles(crystal::Crystal, hkl::Tuple{Int, Int, Int}; dz_tol::Float64=0.8) -> Vector{NamedTuple}

Calculate sub-layer charges, average z-heights, layer dipole vectors mu_layer, and out-of-plane
dipole components mu_z for all sub-layers along the (h,k,l) surface normal in the crystal.
"""
function layer_dipoles(crystal::Crystal, hkl::Tuple{Int, Int, Int}; dz_tol::Float64=0.8)
    if !has_charges(crystal) || iszero(crystal.charges.q)
        throw(
            ArgumentError(
                "Crystal has no atomic charges assigned. Assign charges before calculating layer dipoles.",
            ),
        )
    end
    geom = precompute_slab_geometry(crystal, hkl)
    coords_cart = Cart(crystal.atoms.coords, crystal.box).x
    coords_rot = geom.R * coords_cart
    charges = crystal.charges.q

    z_coords = coords_rot[3, :]
    perm = sortperm(z_coords)

    clusters = Vector{Vector{Int}}()
    current = [perm[1]]
    for i in 2:length(perm)
        idx = perm[i]
        if z_coords[idx] - z_coords[current[end]] <= dz_tol
            push!(current, idx)
        else
            push!(clusters, current)
            current = [idx]
        end
    end
    push!(clusters, current)

    res = NamedTuple{
        (:z_avg, :charge, :mu_layer, :mu_z),
        Tuple{Float64, Float64, Vector{Float64}, Float64},
    }[]
    for cl in clusters
        q_l = sum(charges[idx] for idx in cl)
        z_avg = sum(z_coords[idx] for idx in cl) / length(cl)
        mu_l = vec(
            sum(charges[idx] .* (coords_rot[:, idx] .- [0.0, 0.0, z_avg]) for idx in cl)
        )
        push!(res, (z_avg=z_avg, charge=q_l, mu_layer=mu_l, mu_z=mu_l[3]))
    end

    return res
end

"""
    tasker_type(crystal::Crystal, hkl::Tuple{Int, Int, Int}; tol::Float64=1e-3, dz_tol::Float64=0.8) -> Symbol

Classify a surface plane into Tasker surface categories (Tasker, J. Phys. C, 1979):
- `:Type_I`: Neutral atomic/molecular layers parallel to surface (q_layer == 0 and mu_z == 0 for all layers).
- `:Type_II`: Charged atomic layers (q_layer != 0 or mu_z != 0), but symmetric layer stacking yields zero net surface dipole (sum mu_z == 0).
- `:Type_III`: Charged atomic layers with net out-of-plane electric dipole moment per repeat unit (sum mu_z != 0, polar surface).
"""
function tasker_type(
    crystal::Crystal, hkl::Tuple{Int, Int, Int}; tol::Float64=1e-3, dz_tol::Float64=0.8
)
    if !has_charges(crystal) || iszero(crystal.charges.q)
        slabs = make_slab(crystal, hkl; vacuum=10.0, layers=1)
        return is_polar_surface(slabs[1]; tol=tol) ? :Type_III : :Type_I
    end

    l_data = layer_dipoles(crystal, hkl; dz_tol=dz_tol)
    all_neutral = all(abs(ld.charge) <= tol && abs(ld.mu_z) <= tol for ld in l_data)

    if all_neutral
        return :Type_I
    end

    sum_mu_z = sum(ld.mu_z for ld in l_data)
    if abs(sum_mu_z) <= tol
        return :Type_II
    else
        return :Type_III
    end
end
