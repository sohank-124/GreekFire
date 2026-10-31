# Same schedule/accounting as the corrected ensemble; no duplicated simulator.
prepend!(ARGS, ["validate", "--prediction-mode=momentum"])
include("main.jl")
