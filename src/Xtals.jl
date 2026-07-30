module Xtals

using AtomsBase,
    Bio3DView,
    CSV,
    DataFrames,
    Graphs,
    LinearAlgebra,
    MetaGraphs,
    PeriodicTable,
    Printf,
    StaticArrays,
    Unitful
    PeriodicTable
using PrecompileSignatures: @precompile_signatures

# global variable dictionary
global rc = Dict{Symbol, Any}()
rc[:paths] = Dict{Symbol, String}()

include("matter.jl")
include("box.jl")
include("crystal.jl")
include("distance.jl")
include("misc.jl")
include("repfactors.jl")
include("cpk_colors.jl")
include("covalent_radii.jl")
include("bonds.jl")
include("slab.jl")

rc[:atomic_masses] = Dict([(Symbol(elem.symbol), ustrip(elem.atomic_mass)) for elem in elements[1:118]])
rc[:cpk_colors] = DEFAULT_CPK_COLORS
rc[:covalent_radii] = DEFAULT_COVALENT_RADII
DEFAULT_BONDING_RULES = bondingrules()
rc[:bonding_rules] = DEFAULT_BONDING_RULES

function __init__()
    # create path entries in global dictionary
    rc[:paths][:crystals] = ""
    rc[:paths][:data] = ""
    # set paths to data and crystals relative to pwd() at import
    return set_paths(joinpath(pwd(), "data"); no_warn=true)
end

export
    # Xtals.jl
    rc,

    # matter.jl
    Coords,
    Frac,
    Cart,
    Atoms,
    Charges,
    wrap!,
    neutral,
    net_charge,
    translate_by!,

    # box.jl
    Box,
    replicate,
    unit_cube,
    write_vtk,
    inside,

    # distance.jl
    nearest_image!,
    distance,
    overlap,
    remove_duplicates,
    pairwise_distances,

    # misc.jl
    read_xyz,
    write_xyz,
    read_mol,
    write_mol2,
    assert_P1_symmetry,
    set_paths,
    view_crystal,

    # crystal.jl
    Crystal,
    strip_numbers_from_atom_labels!,
    assign_charges,
    empirical_formula,
    molecular_weight,
    crystal_density,
    write_cif,
    has_charges,
    apply_symmetry_operations,
    write_cssr,
    rename_xtal,
    # AtomsBase things also from crystal
    position,
    velocity,
    cell_vectors,
    periodicity,
    species,
    chemical_formula,

    # bonds.jl
    infer_bonds!,
    write_bond_information,
    BondingRule,
    bond_sanity_check,
    remove_bonds!,
    read_bonding_rules,
    write_bonding_rules,
    add_bonding_rules,
    drop_cross_pb_bonds!,
    bondingrules,
    bond_angle,
    view_bonding_rules,
    bond_length,

    # slab.jl
    make_slab,
    make_slabs,
    surface_basis,
    get_terminations,
    detect_molecules,
    get_bond_vector,
    calculate_bond_vectors!,
    clear_vectors!,
    bond_distance,
    infer_bonds

@precompile_signatures(Xtals)

end # module Xtals
