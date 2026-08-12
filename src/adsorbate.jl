using LinearAlgebra
using StaticArrays
using Graphs
using Random

# =========================================================================================
# 1. FORCE FIELD DATA STRUCTURES & UFF PARAMETERS
# =========================================================================================

"""
    UFFParameters(sigma, epsilon, r_cov)

Data container holding Universal Force Field (UFF) Lennard-Jones parameters and covalent radius:
- `sigma::Float64`: Finite distance in Ångströms at which interatomic potential is zero (σ).
- `epsilon::Float64`: Well depth of interaction potential in kcal/mol (ε).
- `r_cov::Float64`: Covalent radius in Ångströms for physical contact distance evaluation.
"""
struct UFFParameters
    sigma::Float64
    epsilon::Float64
    r_cov::Float64
end

"""
    uff_parameters(species::Symbol) -> UFFParameters

Look up Universal Force Field (UFF) parameters (Rappe et al., JACS 1992) for any chemical element.
Provides a universal fallback (σ = 3.5 Å, ε = 0.10 kcal/mol, r_cov = 1.20 Å) for unlisted species.
"""
function uff_parameters(species::Symbol)
    table = Dict{Symbol, UFFParameters}(
        :H => UFFParameters(2.571, 0.044, 0.31),
        :He => UFFParameters(2.104, 0.056, 0.28),
        :Li => UFFParameters(2.184, 0.025, 1.28),
        :Be => UFFParameters(2.446, 0.085, 0.96),
        :B => UFFParameters(3.638, 0.180, 0.84),
        :C => UFFParameters(3.431, 0.105, 0.76),
        :N => UFFParameters(3.260, 0.069, 0.71),
        :O => UFFParameters(3.118, 0.060, 0.66),
        :F => UFFParameters(3.093, 0.050, 0.57),
        :Ne => UFFParameters(3.243, 0.042, 0.58),
        :Na => UFFParameters(2.658, 0.030, 1.66),
        :Mg => UFFParameters(2.691, 0.111, 1.41),
        :Al => UFFParameters(4.008, 0.505, 1.21),
        :Si => UFFParameters(3.826, 0.402, 1.11),
        :P => UFFParameters(3.695, 0.305, 1.07),
        :S => UFFParameters(3.595, 0.274, 1.05),
        :Cl => UFFParameters(3.517, 0.227, 1.02),
        :Ar => UFFParameters(3.468, 0.238, 1.06),
        :K => UFFParameters(3.396, 0.035, 2.03),
        :Ca => UFFParameters(3.028, 0.238, 1.76),
        :Fe => UFFParameters(2.594, 0.013, 1.32),
        :Co => UFFParameters(2.559, 0.014, 1.26),
        :Ni => UFFParameters(2.525, 0.015, 1.24),
        :Cu => UFFParameters(3.114, 0.005, 1.32),
        :Zn => UFFParameters(2.462, 0.124, 1.22),
        :Br => UFFParameters(3.732, 0.251, 1.20),
        :I => UFFParameters(4.009, 0.339, 1.39),
    )
    return get(table, species, UFFParameters(3.500, 0.100, 1.20))
end

# =========================================================================================
# 2. HIGH-PERFORMANCE CONTIGUOUS DATA CACHES
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

# Constructor to extract substrate data from an initial Crystal slab
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

# Append newly deposited adsorbate atom into the substrate cache for multi-layer stacking
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
                sum(c_cart[1, :])/n, sum(c_cart[2, :])/n, sum(c_cart[3, :])/n
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
    end
    return AdsorbateTemplate(species, charges, rel_coords, sigmas, epsilons, radii, edges)
end

# =========================================================================================
# 3. MATHEMATICAL ORIENTATION & PHYSICAL CONTACT DISTANCE FUNCTIONS
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
        1.0 - 2.0*(q2^2 + q3^2),
        2.0*(q1*q2 + q0*q3),
        2.0*(q1*q3 - q0*q2),
        2.0*(q1*q2 - q0*q3),
        1.0 - 2.0*(q1^2 + q3^2),
        2.0*(q2*q3 + q0*q1),
        2.0*(q1*q3 + q0*q2),
        2.0*(q2*q3 - q0*q1),
        1.0 - 2.0*(q1^2 + q2^2),
    )
end

"""
    find_adsorption_sites(slab::Crystal, site::Union{Symbol, Tuple{Float64, Float64}}, coverage::Tuple{Int, Int}) -> Vector{Tuple{Float64, Float64}}

Calculate fractional (x, y) adsorption site coordinates on the surface of `slab`.
"""
function find_adsorption_sites(
    slab::Crystal, site::Union{Symbol, Tuple{Float64, Float64}}, coverage::Tuple{Int, Int}
)
    Nx_ads, Ny_ads = coverage
    sites = Tuple{Float64, Float64}[]

    if site isa Tuple{Float64, Float64}
        xf, yf = site
        for i in 0:(Nx_ads - 1), j in 0:(Ny_ads - 1)
            push!(sites, (mod(xf + i / Nx_ads, 1.0), mod(yf + j / Ny_ads, 1.0)))
        end
        return sites
    end

    if site == :grid
        for i in 0:(Nx_ads - 1), j in 0:(Ny_ads - 1)
            push!(sites, ((i + 0.5) / Nx_ads, (j + 0.5) / Ny_ads))
        end
    elseif site == :top
        coords_cart = Cart(slab.atoms.coords, slab.box).x
        z_max_idx = argmax(coords_cart[3, :])
        top_frac = slab.atoms.coords.xf[:, z_max_idx]
        for i in 0:(Nx_ads - 1), j in 0:(Ny_ads - 1)
            push!(
                sites,
                (mod(top_frac[1] + i / Nx_ads, 1.0), mod(top_frac[2] + j / Ny_ads, 1.0)),
            )
        end
    elseif site == :bridge
        coords_cart = Cart(slab.atoms.coords, slab.box).x
        n = slab.atoms.n
        perm = sortperm(coords_cart[3, :]; rev=true)
        idx1 = perm[1]
        idx2 = n >= 2 ? perm[2] : perm[1]
        f1 = slab.atoms.coords.xf[:, idx1]
        f2 = slab.atoms.coords.xf[:, idx2]
        bridge_frac = (f1 .+ f2) ./ 2.0
        for i in 0:(Nx_ads - 1), j in 0:(Ny_ads - 1)
            push!(
                sites,
                (
                    mod(bridge_frac[1] + i / Nx_ads, 1.0),
                    mod(bridge_frac[2] + j / Ny_ads, 1.0),
                ),
            )
        end
    else # :hollow fallback
        coords_cart = Cart(slab.atoms.coords, slab.box).x
        perm = sortperm(coords_cart[3, :]; rev=true)
        top_k = min(4, slab.atoms.n)
        mean_frac = vec(sum(slab.atoms.coords.xf[:, perm[1:top_k]]; dims=2) ./ top_k)
        for i in 0:(Nx_ads - 1), j in 0:(Ny_ads - 1)
            push!(
                sites,
                (mod(mean_frac[1] + i / Nx_ads, 1.0), mod(mean_frac[2] + j / Ny_ads, 1.0)),
            )
        end
    end

    return sites
end

"""
    find_contact_height(sub::SubstrateCache, ads::AdsorbateTemplate,
                         rotated_rel_coords::Vector{SVector{3, Float64}},
                         site_xy_cart::SVector{2, Float64},
                         c_to_f::Matrix{Float64}, f_to_c::Matrix{Float64};
                         d_contact::Union{Nothing, Float64}=nothing) -> Float64

Calculate physical non-overlapping z-contact height z_adsorb for a candidate adsorbate state.

# Mathematical Formulation
Lowers the adsorbate along -z until the minimum distance between any adsorbate atom k and substrate
atom j matches physical contact distance d_target (with 2D periodic boundary conditions in x, y):
    d_target = d_contact !== nothing ? d_contact : (r_k^cov + r_j^cov + 0.4 Å)
    z_req = z_slab,j + sqrt(d_target^2 - r_xy,ij^2) - r_z_rel,k
    z_adsorb = max_ij (z_req)
"""
function find_contact_height(
    sub::SubstrateCache,
    ads::AdsorbateTemplate,
    rotated_rel_coords::Vector{SVector{3, Float64}},
    site_xy_cart::SVector{2, Float64},
    c_to_f::Matrix{Float64},
    f_to_c::Matrix{Float64};
    d_contact::Union{Nothing, Float64}=nothing,
)
    n_sub = length(sub.coords)
    n_ads = length(ads.species)
    z_contact_max = -Inf

    for k in 1:n_ads
        rx_k = site_xy_cart[1] + rotated_rel_coords[k][1]
        ry_k = site_xy_cart[2] + rotated_rel_coords[k][2]
        rz_k = rotated_rel_coords[k][3]

        r_ads_k = ads.radii[k]

        for j in 1:n_sub
            # Step 1: Compute lateral Cartesian displacement vector in xy
            dx_cart = rx_k - sub.coords[j][1]
            dy_cart = ry_k - sub.coords[j][2]

            # Step 2: Apply 2D Periodic Boundary Conditions (PBC) in x, y
            d_frac = c_to_f * [dx_cart, dy_cart, 0.0]
            d_frac[1] -= round(d_frac[1])
            d_frac[2] -= round(d_frac[2])
            d_frac[3] = 0.0
            d_cart_pbc = f_to_c * d_frac

            r_xy_sq = d_cart_pbc[1]^2 + d_cart_pbc[2]^2

            # Step 3: Target contact distance evaluation
            target_d = d_contact !== nothing ? d_contact : (r_ads_k + sub.radii[j] + 0.4)

            # Step 4: If lateral distance is within interaction sphere, solve for z_req
            if r_xy_sq < target_d^2
                dz_needed = sqrt(target_d^2 - r_xy_sq)
                z_req = sub.coords[j][3] + dz_needed - rz_k
                if z_req > z_contact_max
                    z_contact_max = z_req
                end
            end
        end
    end

    # Fallback for empty/isolated substrate
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
candidate and substrate atoms.

# Mathematical Energy Model
1. **Lennard-Jones (12-6) Potential with Lorentz-Berthelot Mixing**:
       σ_ij = (σ_i + σ_j) / 2,   ε_ij = sqrt(ε_i * ε_j)
       E_LJ = sum 4 * ε_ij * [ (σ_ij / r_ij)^12 - (σ_ij / r_ij)^6 ]
2. **Coulomb Electrostatics with Polynomial Cutoff Damping**:
       E_Coulomb = sum [ (C_e * q_i * q_j) / r_ij ] * (1 - r_ij^2 / r_cut^2)^2
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

    e_total = 0.0

    for k in 1:n_ads
        q_k = ads.charges[k]
        sig_k = ads.sigmas[k]
        eps_k = ads.epsilons[k]
        r_k = ads_placed_coords[k]

        for j in 1:n_sub
            # Step 1: Pairwise 3D Cartesian displacement with 2D PBC in xy
            dx_cart = r_k[1] - sub.coords[j][1]
            dy_cart = r_k[2] - sub.coords[j][2]
            dz_cart = r_k[3] - sub.coords[j][3]

            d_frac = c_to_f * [dx_cart, dy_cart, dz_cart]
            d_frac[1] -= round(d_frac[1])
            d_frac[2] -= round(d_frac[2])
            d_cart_pbc = f_to_c * d_frac

            r_sq = d_cart_pbc[1]^2 + d_cart_pbc[2]^2 + d_cart_pbc[3]^2

            # Step 2: Cutoff check r_sq < r_cut^2
            if r_sq < r_cut_sq && r_sq > 0.1
                r = sqrt(r_sq)

                # Lorentz-Berthelot mixing rules
                sig_ij = (sig_k + sub.sigmas[j]) / 2.0
                eps_ij = sqrt(eps_k * sub.epsilons[j])

                sr6 = (sig_ij / r)^6
                e_lj = 4.0 * eps_ij * (sr6^2 - sr6)

                # Smooth polynomial cutoff damping for electrostatics
                cutoff_damp = (1.0 - r_sq / r_cut_sq)^2
                e_coulomb = (ke * q_k * sub.charges[j] / r) * cutoff_damp

                e_total += e_lj + e_coulomb
            end
        end
    end

    return e_total
end

# =========================================================================================
# 5. MULTITHREADED MONTE CARLO CANDIDATE SAMPLING
# =========================================================================================

"""
    sample_adsorbate_candidate(sub::SubstrateCache, ads::AdsorbateTemplate,
                                site_xy_cart::SVector{2, Float64},
                                c_to_f::Matrix{Float64}, f_to_c::Matrix{Float64};
                                n_trials::Int=50, d_contact::Union{Nothing, Float64}=nothing,
                                selector::Symbol=:energy, temperature::Float64=300.0) -> Tuple{Vector{SVector{3, Float64}}, Float64}

Executes parallel candidate sampling over n_trials random 3D orientation candidates for an adsorbate molecule
using `Threads.@threads`. Automatically selects the candidate with minimum interaction energy E_min.
"""
function sample_adsorbate_candidate(
    sub::SubstrateCache,
    ads::AdsorbateTemplate,
    site_xy_cart::SVector{2, Float64},
    c_to_f::Matrix{Float64},
    f_to_c::Matrix{Float64};
    n_trials::Int=50,
    d_contact::Union{Nothing, Float64}=nothing,
    selector::Symbol=:energy,
    temperature::Float64=300.0,
)
    n_ads = length(ads.species)
    results = Vector{Tuple{Vector{SVector{3, Float64}}, Float64}}(undef, n_trials)

    # Parallel execution over independent trial candidates via Threads.@threads
    Threads.@threads for t in 1:n_trials
        # 1. Random 3D orientation rotation matrix R in SO(3)
        R_rand = random_rotation_matrix()
        rot_coords = [R_rand * r for r in ads.rel_coords]

        # 2. Downward vertical pulling to physical contact height
        z_adsorb = find_contact_height(
            sub, ads, rot_coords, site_xy_cart, c_to_f, f_to_c; d_contact=d_contact
        )

        # 3. Position candidate in 3D space
        placed_coords = [
            SVector{3, Float64}(
                site_xy_cart[1] + rot_coords[k][1],
                site_xy_cart[2] + rot_coords[k][2],
                z_adsorb + rot_coords[k][3],
            ) for k in 1:n_ads
        ]

        # 4. Energy / Fitness evaluation
        metric_val = 0.0
        if selector == :energy
            metric_val = evaluate_interaction_energy(
                sub, ads, placed_coords, c_to_f, f_to_c
            )
        else # :contact
            metric_val = z_adsorb
        end

        results[t] = (placed_coords, metric_val)
    end

    # Select optimal candidate trial with minimum metric value (lowest energy or closest contact)
    best_idx = argmin([r[2] for r in results])
    return results[best_idx]
end

# =========================================================================================
# 6. TOP-LEVEL USER API (make_adsorbate_slab & add_adsorbate!)
# =========================================================================================

"""
    adsorbed_slab = make_adsorbate_slab(slab::Crystal, adsorbate::Union{Crystal, AbstractString};
                                        coverage::Tuple{Int, Int}=(1, 1),
                                        site::Union{Symbol, Tuple{Float64, Float64}}=:grid,
                                        d_contact::Union{Nothing, Float64}=nothing,
                                        n_layers::Union{Nothing, Int}=nothing,
                                        layer_height::Union{Nothing, Float64}=nothing,
                                        n_trials::Int=50,
                                        selector::Symbol=:energy,
                                        temperature::Float64=300.0,
                                        vacuum::Float64=10.0,
                                        orthogonalize_z::Bool=true) -> Crystal

Deposit single- or multi-layer adsorbates onto a surface slab using a Monte Carlo trial search
and UFF force field interaction energy evaluation.

# Arguments & Keywords
- `slab::Crystal`: Base surface slab structure.
- `adsorbate::Union{Crystal, AbstractString}`: Molecule `Crystal` or file path string (`"water.xyz"`, `"molecule.mol"`).
- `coverage::Tuple{Int, Int}`: Adsorbate replication grid (Nx, Ny) per surface cell.
- `site::Union{Symbol, Tuple{Float64, Float64}}`: Adsorption site mode (`:grid`, `:top`, `:bridge`, `:hollow`, or `(x_frac, y_frac)`).
- `d_contact::Union{Nothing, Float64}`: Target physical contact distance in Ångströms.
- `n_layers::Union{Nothing, Int}`: Target discrete layer count.
- `layer_height::Union{Nothing, Float64}`: Target adsorbate layer film height in Ångströms (e.g. `10.0` Å).
- `n_trials::Int`: Number of random Monte Carlo orientation trial candidates per molecule.
- `selector::Symbol`: Candidate selection mode (`:energy` or `:contact`).
- `vacuum::Float64`: Vacuum gap padding in Ångströms.
"""
function make_adsorbate_slab(
    slab::Crystal,
    adsorbate::Union{Crystal, AbstractString};
    coverage::Tuple{Int, Int}=(1, 1),
    site::Union{Symbol, Tuple{Float64, Float64}}=:grid,
    d_contact::Union{Nothing, Float64}=nothing,
    n_layers::Union{Nothing, Int}=nothing,
    layer_height::Union{Nothing, Float64}=nothing,
    n_trials::Int=50,
    selector::Symbol=:energy,
    temperature::Float64=300.0,
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
    if n_layers !== nothing && n_layers < 1
        throw(ArgumentError("n_layers must be >= 1, got $n_layers"))
    end

    # Step 1: Prepare adsorbate template and initialize substrate cache
    ads = prepare_adsorbate_template(adsorbate)
    sub_cache = SubstrateCache(slab)

    # Step 2: Determine adsorbate layer count N_layers from layer_height target or n_layers keyword
    ads_z_extent = if isempty(ads.rel_coords)
        1.0
    else
        (maximum(c[3] for c in ads.rel_coords) - minimum(c[3] for c in ads.rel_coords))
    end
    ads_layer_spacing = max(2.2, ads_z_extent + (d_contact !== nothing ? d_contact : 2.0))

    N_layers = 1
    if layer_height !== nothing && layer_height > 0.0
        N_layers = max(1, ceil(Int, layer_height / ads_layer_spacing))
    elseif n_layers !== nothing && n_layers >= 1
        N_layers = n_layers
    end

    # Step 3: Find surface adsorption site fractional positions
    site_fracs = find_adsorption_sites(slab, site, coverage)
    c_to_f = slab.box.c_to_f
    f_to_c = slab.box.f_to_c

    slab_cart = Cart(slab.atoms.coords, slab.box).x
    n_slab_atoms = slab.atoms.n

    all_species = Symbol[slab.atoms.species...]
    all_charges =
        has_charges(slab) ? Float64[slab.charges.q...] : zeros(Float64, n_slab_atoms)
    all_coords_cart = Vector{Float64}[slab_cart[:, i] for i in 1:n_slab_atoms]
    ads_edges_all = Tuple{Int, Int}[]

    # Step 4: Depositing adsorbate layers k = 1..N_layers
    for layer_k in 1:N_layers
        for s_frac in site_fracs
            site_cart_3d = f_to_c * [s_frac[1], s_frac[2], 0.0]
            site_xy_cart = SVector{2, Float64}(site_cart_3d[1], site_cart_3d[2])

            # Sample optimal candidate configuration via Monte Carlo trial search
            placed_coords, _ = sample_adsorbate_candidate(
                sub_cache,
                ads,
                site_xy_cart,
                c_to_f,
                f_to_c;
                n_trials=n_trials,
                d_contact=d_contact,
                selector=selector,
                temperature=temperature,
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
        end
    end

    # Step 5: Construct final cell box with expanded z-height to fit slab + adsorbate film + vacuum padding
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

    adsorbed_name = "$(slab.name)_adsorbed_L$(N_layers)"
    adsorbed_crystal = Crystal(adsorbed_name, new_box, final_atoms, final_charges)

    # Step 6: Preserve slab and adsorbate bond graphs
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
    add_adsorbate!(slab::Crystal, adsorbate::Union{Symbol, Crystal, AbstractString}; kwargs...)

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
