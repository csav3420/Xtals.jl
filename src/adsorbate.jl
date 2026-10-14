# =========================================================================================
# MODULE: adsorbate.jl
# Surface Adsorbate & Solvation Overlayer Film Deposition Engine for Xtals.jl
#
# Provides pure continuous 2D random Monte Carlo candidate sampling and Universal Force Field
# (UFF) Lennard-Jones (12-6) + Damped Coulomb electrostatic interaction energy evaluation.
# =========================================================================================

using Random

# =========================================================================================
# 1. UFF FORCE FIELD PARAMETER TABLE
# =========================================================================================

"""
    UFFParameters

Holds Universal Force Field (UFF) parameters:
- `sigma::Float64`: Lennard-Jones equilibrium collision diameter σ in Ångströms.
- `epsilon::Float64`: Lennard-Jones potential well depth ε in kcal/mol.
- `r_cov::Float64`: Covalent radius in Ångströms for physical contact distance calculations.
"""
struct UFFParameters
    sigma::Float64
    epsilon::Float64
    r_cov::Float64
end

"""
    uff_parameters(species::Symbol) -> UFFParameters

Lookup table for UFF parameters and covalent radii.
"""
function uff_parameters(species::Symbol)
    table = Dict{Symbol, UFFParameters}(
        :H => UFFParameters(2.571, 0.044, 0.31),
        :C => UFFParameters(3.431, 0.105, 0.76),
        :N => UFFParameters(3.260, 0.069, 0.71),
        :O => UFFParameters(3.118, 0.060, 0.66),
        :F => UFFParameters(2.997, 0.050, 0.57),
        :Na => UFFParameters(2.658, 0.030, 1.02),
        :Mg => UFFParameters(2.691, 0.111, 1.39),
        :Al => UFFParameters(4.008, 0.505, 1.24),
        :Si => UFFParameters(3.826, 0.402, 1.11),
        :P => UFFParameters(3.695, 0.305, 1.07),
        :S => UFFParameters(3.595, 0.274, 1.05),
        :Cl => UFFParameters(3.517, 0.227, 1.02),
        :K => UFFParameters(3.396, 0.035, 1.38),
        :Ca => UFFParameters(3.028, 0.238, 1.71),
        :Ti => UFFParameters(3.175, 0.017, 1.36),
        :Fe => UFFParameters(2.594, 0.013, 1.26),
        :Cu => UFFParameters(3.114, 0.005, 1.32),
        :Zn => UFFParameters(2.763, 0.124, 1.22),
        :Br => UFFParameters(3.856, 0.251, 1.20),
        :I => UFFParameters(4.150, 0.339, 1.39),
    )

    if haskey(table, species)
        return table[species]
    else
        return UFFParameters(3.400, 0.100, 0.75)
    end
end

# =========================================================================================
# 2. SUBSTRATE CACHE & ADSORBATE TEMPLATE
# =========================================================================================

"""
    SubstrateCache

High-performance contiguous data cache holding 3D coordinates (using stack-allocated `SVector`),
atomic charges, UFF parameters (σ, ε), and covalent radii for substrate (slab + placed adsorbates).
Avoids expensive Crystal queries during inner Monte Carlo sampling loops.
"""
struct SubstrateCache
    coords::Vector{SVector{3, Float64}}
    charges::Vector{Float64}
    sigmas::Vector{Float64}
    epsilons::Vector{Float64}
    radii::Vector{Float64}
end

function SubstrateCache(crystal::Crystal)
    c_cart = Cart(crystal.atoms.coords, crystal.box).x
    n = crystal.atoms.n
    coords = Vector{SVector{3, Float64}}(undef, n)
    charges = has_charges(crystal) ? Float64[crystal.charges.q...] : zeros(Float64, n)
    sigmas = Vector{Float64}(undef, n)
    epsilons = Vector{Float64}(undef, n)
    radii = Vector{Float64}(undef, n)

    for i in 1:n
        coords[i] = SVector{3, Float64}(c_cart[1, i], c_cart[2, i], c_cart[3, i])
        uff = uff_parameters(crystal.atoms.species[i])
        sigmas[i] = uff.sigma
        epsilons[i] = uff.epsilon
        radii[i] = uff.r_cov
    end

    return SubstrateCache(coords, charges, sigmas, epsilons, radii)
end

function push_atom!(
    cache::SubstrateCache, coord::SVector{3, Float64}, charge::Float64, species::Symbol
)
    push!(cache.coords, coord)
    push!(cache.charges, charge)
    uff = uff_parameters(species)
    push!(cache.sigmas, uff.sigma)
    push!(cache.epsilons, uff.epsilon)
    return push!(cache.radii, uff.r_cov)
end

"""
    AdsorbateTemplate

Pre-computed template holding species, charges, relative 3D centroid coordinates,
UFF parameters, and intra-molecular bonding edges for an adsorbate molecule.
"""
struct AdsorbateTemplate
    species::Vector{Symbol}
    charges::Vector{Float64}
    rel_coords::Vector{SVector{3, Float64}}
    sigmas::Vector{Float64}
    epsilons::Vector{Float64}
    radii::Vector{Float64}
    edges::Vector{Tuple{Int, Int}}
end

# Prepare adsorbate template from a Crystal molecule or file path
function prepare_adsorbate_template(adsorbate::Union{Crystal, AbstractString})
    if adsorbate isa AbstractString
        return prepare_adsorbate_template(read_adsorbate(adsorbate))
    else
        mols = detect_molecules(adsorbate)
        if !isempty(mols)
            m = mols[1]
            species = m.species
            charges = m.charges
            n = length(species)
            rel_coords = [
                SVector{3, Float64}(
                    m.relative_coords_cart[1, i],
                    m.relative_coords_cart[2, i],
                    m.relative_coords_cart[3, i],
                ) for i in 1:n
            ]
            edges = m.edges
        else
            species = adsorbate.atoms.species
            n = length(species)
            charges = has_charges(adsorbate) ? adsorbate.charges.q : zeros(Float64, n)
            c_cart = Cart(adsorbate.atoms.coords, adsorbate.box).x
            centroid = SVector{3, Float64}(
                sum(c_cart[1, :]) / n, sum(c_cart[2, :]) / n, sum(c_cart[3, :]) / n
            )
            rel_coords = [
                SVector{3, Float64}(c_cart[1, i], c_cart[2, i], c_cart[3, i]) - centroid for
                i in 1:n
            ]
            edges = Tuple{Int, Int}[]
        end
        sigmas = Vector{Float64}(undef, n)
        epsilons = Vector{Float64}(undef, n)
        radii = Vector{Float64}(undef, n)
        for i in 1:n
            uff = uff_parameters(species[i])
            sigmas[i] = uff.sigma
            epsilons[i] = uff.epsilon
            radii[i] = uff.r_cov
        end
        return AdsorbateTemplate(
            species, charges, rel_coords, sigmas, epsilons, radii, edges
        )
    end
end

# =========================================================================================
# 3. QUATERNION ROTATION & CONTACT HEIGHT CALCULATIONS
# =========================================================================================

"""
    random_rotation_matrix() -> SMatrix{3, 3, Float64, 9}

Generate a uniform random 3D rotation matrix R in SO(3) using random unit quaternions:
    q = (q0, q1, q2, q3) where q0^2 + q1^2 + q2^2 + q3^2 = 1
This provides mathematically uniform 3D spatial rotation sampling without gimbal lock bias.
"""
function random_rotation_matrix()
    u1, u2, u3 = rand(), rand(), rand()
    q0 = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    q1 = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    q2 = sqrt(u1) * sin(2.0 * pi * u3)
    q3 = sqrt(u1) * cos(2.0 * pi * u3)

    return SMatrix{3, 3, Float64, 9}(
        1.0 - 2.0 * (q2^2 + q3^2),
        2.0 * (q1 * q2 + q0 * q3),
        2.0 * (q1 * q3 - q0 * q2),
        2.0 * (q1 * q2 - q0 * q3),
        1.0 - 2.0 * (q1^2 + q3^2),
        2.0 * (q2 * q3 + q0 * q1),
        2.0 * (q1 * q3 + q0 * q2),
        2.0 * (q2 * q3 - q0 * q1),
        1.0 - 2.0 * (q1^2 + q2^2),
    )
end

"""
    find_contact_height(sub::SubstrateCache, ads::AdsorbateTemplate,
                         rotated_rel_coords::Vector{SVector{3, Float64}},
                         cand_xy_cart::SVector{2, Float64},
                         c_to_f::Matrix{Float64}, f_to_c::Matrix{Float64};
                         d_contact::Union{Nothing, Float64}=nothing) -> Float64

Calculate physical non-overlapping z-contact height z_adsorb for a candidate adsorbate state.
Lowers the adsorbate along -z until the minimum distance between any adsorbate atom k and substrate
atom j matches physical van der Waals contact distance d_target (taking 2D periodic boundary conditions into account).
"""
function find_contact_height(
    sub::SubstrateCache,
    ads::AdsorbateTemplate,
    rotated_rel_coords::Vector{SVector{3, Float64}},
    cand_xy_cart::SVector{2, Float64},
    c_to_f::Matrix{Float64},
    f_to_c::Matrix{Float64};
    d_contact::Union{Nothing, Float64}=nothing,
)
    n_sub = length(sub.coords)
    n_ads = length(ads.species)
    z_contact_max = -Inf

    # Extract 2D surface basis matrices to prevent 3D tilt coordinate mixing
    f_to_c_2d = f_to_c[1:2, 1:2]
    c_to_f_2d = inv(f_to_c_2d)

    for k in 1:n_ads
        rx_k = cand_xy_cart[1] + rotated_rel_coords[k][1]
        ry_k = cand_xy_cart[2] + rotated_rel_coords[k][2]
        rz_k = rotated_rel_coords[k][3]

        sig_k = ads.sigmas[k]

        for j in 1:n_sub
            dx_cart = rx_k - sub.coords[j][1]
            dy_cart = ry_k - sub.coords[j][2]

            df_x = c_to_f_2d[1, 1] * dx_cart + c_to_f_2d[1, 2] * dy_cart
            df_y = c_to_f_2d[2, 1] * dx_cart + c_to_f_2d[2, 2] * dy_cart

            df_x -= round(df_x)
            df_y -= round(df_y)

            dx_pbc = f_to_c_2d[1, 1] * df_x + f_to_c_2d[1, 2] * df_y
            dy_pbc = f_to_c_2d[2, 1] * df_x + f_to_c_2d[2, 2] * df_y

            r_xy_sq = dx_pbc^2 + dy_pbc^2

            # Use physical van der Waals UFF contact threshold (sigma_ij * 0.85)
            target_d =
                d_contact !== nothing ? d_contact : (0.85 * (sig_k + sub.sigmas[j]) / 2.0)

            if r_xy_sq < target_d^2
                dz_needed = sqrt(target_d^2 - r_xy_sq)
                z_req = sub.coords[j][3] + dz_needed - rz_k
                if z_req > z_contact_max
                    z_contact_max = z_req
                end
            end
        end
    end

    if isinf(z_contact_max)
        z_sub_max = maximum(c[3] for c in sub.coords)
        z_contact_max = z_sub_max + (d_contact !== nothing ? d_contact : 2.5)
    end

    return z_contact_max
end

# =========================================================================================
# 4. ENERGY EVALUATION (UFF LENNARD-JONES + ELECTROSTATIC COULOMB)
# =========================================================================================

"""
    evaluate_interaction_energy(sub::SubstrateCache, ads::AdsorbateTemplate,
                                 ads_placed_coords::Vector{SVector{3, Float64}},
                                 c_to_f::Matrix{Float64}, f_to_c::Matrix{Float64};
                                 r_cut::Float64=10.0) -> Float64

Calculate pairwise interaction energy E_total = E_LJ + E_Coulomb (kcal/mol) between an adsorbate
candidate and substrate atoms using 2D lateral periodic boundary conditions.
"""
function evaluate_interaction_energy(
    sub::SubstrateCache,
    ads::AdsorbateTemplate,
    ads_placed_coords::Vector{SVector{3, Float64}},
    c_to_f::Matrix{Float64},
    f_to_c::Matrix{Float64};
    r_cut::Float64=10.0,
)
    n_sub = length(sub.coords)
    n_ads = length(ads.species)
    r_cut_sq = r_cut^2
    ke = 1389.354 # Coulomb constant C_e in kcal/mol * Å / e^2

    # Extract 2D surface basis matrices to prevent 3D tilt coordinate mixing
    f_to_c_2d = f_to_c[1:2, 1:2]
    c_to_f_2d = inv(f_to_c_2d)

    e_total = 0.0

    for k in 1:n_ads
        q_k = ads.charges[k]
        sig_k = ads.sigmas[k]
        eps_k = ads.epsilons[k]
        r_k = ads_placed_coords[k]

        for j in 1:n_sub
            dx_cart = r_k[1] - sub.coords[j][1]
            dy_cart = r_k[2] - sub.coords[j][2]
            dz_cart = r_k[3] - sub.coords[j][3]

            df_x = c_to_f_2d[1, 1] * dx_cart + c_to_f_2d[1, 2] * dy_cart
            df_y = c_to_f_2d[2, 1] * dx_cart + c_to_f_2d[2, 2] * dy_cart

            df_x -= round(df_x)
            df_y -= round(df_y)

            dx_pbc = f_to_c_2d[1, 1] * df_x + f_to_c_2d[1, 2] * df_y
            dy_pbc = f_to_c_2d[2, 1] * df_x + f_to_c_2d[2, 2] * df_y

            r_sq = dx_pbc^2 + dy_pbc^2 + dz_cart^2

            if r_sq < r_cut_sq && r_sq > 0.1
                r = sqrt(r_sq)

                sig_ij = (sig_k + sub.sigmas[j]) / 2.0
                eps_ij = sqrt(eps_k * sub.epsilons[j])

                sr6 = (sig_ij / r)^6
                e_lj = 4.0 * eps_ij * (sr6^2 - sr6)

                cutoff_damp = (1.0 - r_sq / r_cut_sq)^2
                e_coulomb = (ke * q_k * sub.charges[j] / r) * cutoff_damp

                e_total += e_lj + e_coulomb
            end
        end
    end

    return e_total
end

# =========================================================================================
# 5. MULTITHREADED CONTINUOUS 2D MONTE CARLO CANDIDATE SAMPLING
# =========================================================================================

"""
    sample_adsorbate_candidate(sub::SubstrateCache, ads::AdsorbateTemplate,
                                c_to_f::Matrix{Float64}, f_to_c::Matrix{Float64};
                                n_trials::Int=50, d_contact::Union{Nothing, Float64}=nothing) -> Tuple{Vector{SVector{3, Float64}}, Float64}

Executes parallel candidate sampling over n_trials continuous 2D positions across the unit cell and random
3D unit quaternion orientations using `Threads.@threads`. Automatically selects candidate with minimum interaction energy E_min.
"""
function sample_adsorbate_candidate(
    sub::SubstrateCache,
    ads::AdsorbateTemplate,
    c_to_f::Matrix{Float64},
    f_to_c::Matrix{Float64};
    n_trials::Int=50,
    d_contact::Union{Nothing, Float64}=nothing,
)
    n_ads = length(ads.species)
    results = Vector{Tuple{Vector{SVector{3, Float64}}, Float64}}(undef, n_trials)

    # Parallel execution over independent trial candidates via Threads.@threads
    Threads.@threads for t in 1:n_trials
        # 1. Thread-local random continuous 2D fractional coordinates across full cell [0, 1)^2
        xf = rand()
        yf = rand()
        cand_cart_3d = f_to_c * [xf, yf, 0.0]
        cand_xy_cart = SVector{2, Float64}(cand_cart_3d[1], cand_cart_3d[2])

        # 2. Thread-local random 3D orientation rotation matrix R in SO(3)
        R_rand = random_rotation_matrix()
        rot_coords = [R_rand * r for r in ads.rel_coords]

        # 3. Downward vertical pulling to physical contact height
        z_adsorb = find_contact_height(
            sub, ads, rot_coords, cand_xy_cart, c_to_f, f_to_c; d_contact=d_contact
        )

        # 4. Position candidate in 3D space
        placed_coords = [
            SVector{3, Float64}(
                cand_xy_cart[1] + rot_coords[k][1],
                cand_xy_cart[2] + rot_coords[k][2],
                z_adsorb + rot_coords[k][3],
            ) for k in 1:n_ads
        ]

        # 5. Interaction energy evaluation
        e_tot = evaluate_interaction_energy(sub, ads, placed_coords, c_to_f, f_to_c)

        results[t] = (placed_coords, e_tot)
    end

    # Select candidate trial with minimum interaction energy
    best_idx = argmin([r[2] for r in results])
    return results[best_idx]
end

# =========================================================================================
# 6. TOP-LEVEL USER API (make_adsorbate_slab & add_adsorbate!)
# =========================================================================================

"""
    adsorbed_slab = make_adsorbate_slab(slab::Crystal, adsorbate::Union{Crystal, AbstractString};
                                        n_molecules::Union{Nothing, Int}=nothing,
                                        layer_height::Union{Nothing, Float64}=nothing,
                                        d_contact::Union{Nothing, Float64}=nothing,
                                        n_trials::Int=50,
                                        seed::Union{Nothing, Int}=nothing,
                                        vacuum::Float64=10.0,
                                        orthogonalize_z::Bool=true) -> Crystal

Deposit single- or multi-layer adsorbates smoothly across the full surface cell using continuous
2D Monte Carlo trial sampling and UFF interaction energy minimization.

# Arguments & Keywords
- `slab::Crystal`: Base surface slab structure.
- `adsorbate::Union{Crystal, AbstractString}`: Molecule `Crystal` or file path string (`"water.xyz"`, `"molecule.mol"`).
- `n_molecules::Union{Nothing, Int}`: Target total number of adsorbate molecules to deposit across full cell.
- `layer_height::Union{Nothing, Float64}`: Target adsorbate solvation film height in Ångströms (e.g. `10.0` Å).
- `d_contact::Union{Nothing, Float64}`: Target physical contact distance in Ångströms.
- `n_trials::Int`: Number of random parallel Monte Carlo trial candidates per molecule insertion.
- `seed::Union{Nothing, Int}`: Optional integer seed for 100% reproducible random Monte Carlo sampling.
- `vacuum::Float64`: Vacuum gap padding in Ångströms.
"""
function make_adsorbate_slab(
    slab::Crystal,
    adsorbate::Union{Crystal, AbstractString};
    n_molecules::Union{Nothing, Int}=nothing,
    layer_height::Union{Nothing, Float64}=nothing,
    d_contact::Union{Nothing, Float64}=nothing,
    n_trials::Int=50,
    seed::Union{Nothing, Int}=nothing,
    vacuum::Float64=10.0,
    orthogonalize_z::Bool=true,
)
    # Input validation guard clauses
    if vacuum < 0.0
        throw(ArgumentError("vacuum padding must be non-negative, got $vacuum"))
    end
    if n_trials < 1
        throw(ArgumentError("n_trials must be >= 1, got $n_trials"))
    end
    if layer_height !== nothing && layer_height <= 0.0
        throw(ArgumentError("layer_height must be positive, got $layer_height"))
    end
    if n_molecules !== nothing && n_molecules < 1
        throw(ArgumentError("n_molecules must be >= 1, got $n_molecules"))
    end

    # Reproducible RNG seed initialization if specified
    if seed !== nothing
        Random.seed!(seed)
    end

    # Step 1: Prepare adsorbate template and initialize substrate cache
    ads = prepare_adsorbate_template(adsorbate)
    sub_cache = SubstrateCache(slab)

    c_to_f = slab.box.c_to_f
    f_to_c = slab.box.f_to_c

    slab_cart = Cart(slab.atoms.coords, slab.box).x
    n_slab_atoms = slab.atoms.n
    z_slab_initial_max = maximum(slab_cart[3, :])

    all_species = Symbol[slab.atoms.species...]
    all_charges =
        has_charges(slab) ? Float64[slab.charges.q...] : zeros(Float64, n_slab_atoms)
    all_coords_cart = Vector{Float64}[slab_cart[:, i] for i in 1:n_slab_atoms]
    ads_edges_all = Tuple{Int, Int}[]

    # Step 2: Determine insertion count or height termination criteria
    target_height =
        layer_height !== nothing ? layer_height : (n_molecules === nothing ? 3.0 : Inf)
    max_mols = n_molecules !== nothing ? n_molecules : 1000

    m_inserted = 0

    while m_inserted < max_mols
        # Sample optimal candidate configuration via parallel continuous Monte Carlo trial search
        placed_coords, _ = sample_adsorbate_candidate(
            sub_cache, ads, c_to_f, f_to_c; n_trials=n_trials, d_contact=d_contact
        )

        offset = length(all_species)

        for a in 1:length(ads.species)
            push!(all_species, ads.species[a])
            push!(all_charges, ads.charges[a])
            push!(
                all_coords_cart,
                [placed_coords[a][1], placed_coords[a][2], placed_coords[a][3]],
            )
            push_atom!(sub_cache, placed_coords[a], ads.charges[a], ads.species[a])
        end

        for (u, v) in ads.edges
            push!(ads_edges_all, (offset + u, offset + v))
        end

        m_inserted += 1

        # Check film height termination criteria
        current_max_z = maximum(c[3] for c in all_coords_cart)
        if layer_height !== nothing && (current_max_z - z_slab_initial_max) >= target_height
            break
        elseif layer_height === nothing &&
            n_molecules === nothing &&
            (current_max_z - z_slab_initial_max) >= 3.0
            break
        end
    end

    # Step 3: Construct final cell box with expanded z-height to fit slab + adsorbate film + vacuum padding
    max_z_all = maximum(c[3] for c in all_coords_cart)
    min_z_all = minimum(c[3] for c in all_coords_cart)

    total_slab_ads_z = max_z_all - min_z_all
    new_box_z = total_slab_ads_z + 2.0 * vacuum

    a_vec = slab.box.f_to_c[:, 1]
    b_vec = slab.box.f_to_c[:, 2]

    if orthogonalize_z
        c_vec = [0.0, 0.0, new_box_z]
    else
        c_vec = [slab.box.f_to_c[1, 3], slab.box.f_to_c[2, 3], new_box_z]
    end

    new_f_to_c = [a_vec b_vec c_vec]
    new_box = Box(new_f_to_c)

    coords_mat = hcat(all_coords_cart...)
    z_shift_center = vacuum - min_z_all
    coords_mat[3, :] .+= z_shift_center

    atoms_cart = Cart(coords_mat)
    atoms_frac = Frac(atoms_cart, new_box)
    final_atoms = Atoms(all_species, atoms_frac)
    final_charges = Charges(all_charges, atoms_frac)

    adsorbed_name = "$(slab.name)_adsorbed"
    adsorbed_crystal = Crystal(adsorbed_name, new_box, final_atoms, final_charges)

    # Step 4: Preserve slab and adsorbate bond graphs
    if ne(slab.bonds) > 0
        for e in edges(slab.bonds)
            add_edge!(adsorbed_crystal.bonds, src(e), dst(e))
        end
    end
    for (u, v) in ads_edges_all
        add_edge!(adsorbed_crystal.bonds, u, v)
    end

    return adsorbed_crystal
end

"""
    add_adsorbate!(slab::Crystal, adsorbate::Union{Crystal, AbstractString}; kwargs...)

In-place alias for `make_adsorbate_slab`.
"""
const add_adsorbate! = make_adsorbate_slab

"""
    make_adsorbate_slab(slab::Crystal, filename::AbstractString; kwargs...)

Overload for depositing adsorbates directly from any computational chemistry file (.xyz, .mol, .mol2, .pdb, .sdf).
"""
function make_adsorbate_slab(slab::Crystal, filename::AbstractString; kwargs...)
    return make_adsorbate_slab(slab, read_adsorbate(filename); kwargs...)
end

"""
    read_adsorbate(filename::AbstractString) -> Crystal

Read an adsorbate molecule from disk using `Chemfiles.jl` (.xyz, .mol, .mol2, .pdb, .sdf, .gro, .cif)
and return it as a non-periodic `Crystal` object.
"""
function read_adsorbate(filename::AbstractString)
    return read_crystal_chemfiles(filename)
end
