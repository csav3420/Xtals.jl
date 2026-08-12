using Xtals, LinearAlgebra, Test, BenchmarkTools, Dates

set_paths(joinpath(dirname(pathof(Xtals)), "..", "test", "data"))

@testset "Statistical Performance Benchmarks (BenchmarkTools)" begin
    aspirin = Crystal("aspirin.cif")
    aspirin_chg = assign_charges(aspirin, Dict(:C => 0.1, :H => 0.05, :O => -0.325), Inf)

    tsv_file = joinpath(dirname(pathof(Xtals)), "..", "test", "benchmark_history.tsv")

    # Load last benchmark run metrics for timeline comparison
    function load_last_run(file_path::String)
        last_run = Dict{String, Tuple{Float64, Float64}}()
        if !isfile(file_path)
            return last_run
        end
        lines = readlines(file_path)
        if length(lines) <= 1
            return last_run
        end
        last_line = lines[end]
        parts_last = split(last_line, "\t")
        if length(parts_last) < 5
            return last_run
        end
        last_ts = parts_last[1]

        for line in lines[2:end]
            parts = split(line, "\t")
            if parts[1] == last_ts
                if length(parts) >= 6
                    op = parts[4]
                    t_ms = parse(Float64, parts[5])
                    m_mb = parse(Float64, parts[6])
                    last_run[op] = (t_ms, m_mb)
                elseif length(parts) == 5
                    op = parts[3]
                    t_ms = parse(Float64, parts[4])
                    m_mb = parse(Float64, parts[5])
                    last_run[op] = (t_ms, m_mb)
                end
            end
        end
        return last_run
    end

    prev_run = load_last_run(tsv_file)
    current_ts = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS")
    num_threads = Threads.nthreads()

    if !isfile(tsv_file)
        open(tsv_file, "w") do io
            println(io, "timestamp\tjulia_version\tthreads\toperation\ttime_ms\tmemory_mb")
        end
    end

    println(
        "\n==========================================================================================",
    )
    println(
        "             XTALS BENCHMARKTOOLS STATISTICAL TIMELINE TRACKER (Threads: $(num_threads))        ",
    )
    println(
        "==========================================================================================",
    )
    println(
        "Operation                               | Med Time(ms) | Memory (MB) | Timeline vs Prev Run",
    )
    println(
        "------------------------------------------------------------------------------------------",
    )

    function run_and_log_bt(op_name::String, b)
        tune!(b)
        res = run(b; samples=50, evals=1)

        med_t = median(res)
        time_ms = round(BenchmarkTools.time(med_t) / 1e6, digits=2)
        mem_mb = round(BenchmarkTools.memory(med_t) / (1024.0 * 1024.0), digits=2)

        # Append to TSV log file with thread count
        open(tsv_file, "a") do io
            println(
                io, "$current_ts\t$(VERSION)\t$num_threads\t$op_name\t$time_ms\t$mem_mb"
            )
        end

        change_str = "Initial Run"
        if haskey(prev_run, op_name)
            prev_t, _ = prev_run[op_name]
            diff_t = round(time_ms - prev_t, digits=2)
            pct = prev_t > 0 ? round(diff_t / prev_t * 100.0, digits=1) : 0.0
            sign_str = diff_t > 0 ? "+" : ""
            change_str = "$(sign_str)$(diff_t) ms ($(sign_str)$(pct)%)"
        end

        op_padded = rpad(op_name, 39)
        t_padded = rpad(string(time_ms), 12)
        m_padded = rpad(string(mem_mb), 11)
        println("$op_padded | $t_padded | $m_padded | $change_str")
        @test time_ms >= 0.0 # Valid time measurement assertion
    end

    run_and_log_bt(
        "surface_basis (2, 1, 2)",
        @benchmarkable surface_basis((2, 1, 2); f_to_c=($(aspirin.box.f_to_c)))
    )
    run_and_log_bt("detect_molecules (Aspirin)", @benchmarkable detect_molecules($aspirin))
    run_and_log_bt(
        "get_terminations (0,1,0; unique)",
        @benchmarkable get_terminations($aspirin, (0, 1, 0); unique_only=true)
    )
    run_and_log_bt(
        "make_slab (1,1,0; 1 layer)",
        @benchmarkable make_slab($aspirin, (1, 1, 0); vacuum=10.0, layers=1)
    )
    run_and_log_bt(
        "make_slab (2,1,2; min_size 50x50 Å)",
        @benchmarkable make_slab(
            $aspirin, (2, 1, 2); min_size=(50.0, 50.0), vacuum=10.0, layers=1
        )
    )
    run_and_log_bt(
        "layer_dipoles (1, 0, 0)", @benchmarkable layer_dipoles($aspirin_chg, (1, 0, 0))
    )
    run_and_log_bt(
        "tasker_type (1, 0, 0)", @benchmarkable tasker_type($aspirin_chg, (1, 0, 0))
    )

    println(
        "==========================================================================================",
    )
    println("Benchmark timeline metrics logged to: $tsv_file\n")
end
