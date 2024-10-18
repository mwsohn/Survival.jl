"""
    KaplanMeier{S,T}

An immutable type containing survivor function estimates computed
using the Kaplan-Meier method.
The type has the following fields:

* `events`: An [`EventTable`](@ref) summarizing the times and events
  used to compute the estimates. The time values are of type `T`.
* `survival`: Estimate of the survival probability at each time. Values
  are of type `S`.
* `stderr`: Standard error of the log survivor function at each time.
  Values are of type `S`.

Use `fit(KaplanMeier, ...)` to compute the estimates as `Float64`
values and construct this type.
Alternatively, `fit(KaplanMeier{S}, ...)` may be used to request a
particular value type `S` for the estimates.
"""
struct KaplanMeier{S,T} <: NonparametricEstimator
    events::EventTable{T}
    survival::Vector{S}
    stderr::Vector{S}
end

estimator_eltype(::Type{<:KaplanMeier{S}}) where {S} = S
estimator_eltype(::Type{KaplanMeier}) = Float64

estimator_start(T::Type{<:KaplanMeier}) = oneunit(estimator_eltype(T))
stderr_start(T::Type{<:KaplanMeier}) = zero(estimator_eltype(T))

estimator_update(::Type{<:KaplanMeier}, es, dᵢ, nᵢ) = es * (1 - dᵢ // nᵢ)
stderr_update(::Type{<:KaplanMeier}, gw, dᵢ, nᵢ) = gw + dᵢ // (nᵢ * (nᵢ - dᵢ))

"""
    confint(km::KaplanMeier; level=0.05)

Compute the pointwise log-log transformed confidence intervals for the survivor
function as a vector of tuples.
"""
function StatsAPI.confint(km::KaplanMeier; level::Real=0.05)
    q = quantile(Normal(), 1 - level/2)
    return map(km.survival, km.stderr) do srv, se
        l = log(-log(srv))
        a = q * se / log(srv)
        exp(-exp(l - a)), exp(-exp(l + a))
    end
end

"""
    fit(KaplanMeier, times, status) -> KaplanMeier

Given a vector of times to events and a corresponding vector of indicators that
denote whether each time is an observed event or is right censored, compute the
Kaplan-Meier estimate of the survivor function.
"""
StatsAPI.fit(::Type{KaplanMeier}, times, status)

"""
    fit(KaplanMeier, ets) -> KaplanMeier

Compute the Kaplan-Meier estimate of the survivor function from a vector of
[`EventTime`](@ref) values.
"""
StatsAPI.fit(::Type{KaplanMeier}, ets)

"""
    kaplanmeier(df, event; by = nothing)

Plots Kaplan-Meier estimates.
"""
function kaplanmeier(df, event::EventTime; by::Symbol=nothing)

    plt = nothing
    if by == nothing
        km = fit(KaplanMeier, df[:, event])
        plot(vcat(0, km.events.time), vcat(1.0, km.survival), linetype=:steppost, ylims=(0, 1))
        # return nothing
    else

        kvec = []

        for (i, v) in enumerate(sort(unique(skipmissing(df[:, by]))))
            df2 = filter(x -> !ismissing(x[by]) && x[by] == v, df)
            push!(kvec, fit(KaplanMeier, df2[!, event]))
            if i == 1
                plt = Plots.plot(vcat(0, kvec[1].events.time),
                    vcat(1.0, kvec[1].survival),
                    linetype=:steppost,
                    ylims=(0, 1),
                    xlabel="Analysis time",
                    ylabel="Survival estimate",
                    label=string("$by = ", v))

            else
                Plots.plot!(plt, vcat(0, kvec[i].events.time),
                    vcat(1.0, kvec[i].survival),
                    linetype=:steppost,
                    ylims=(0, 1),
                    label=string("$by = ", v))

            end
        end
    end
    return plt
end


# struct LogRank
#     observed::Vector{Int64}()
#     expected::Vector{Float64}()
#     nobs::Int64
#     dof::Int64
#     chi2::Float64
#     pvalue::Float64
# end

# function Base.show(io::IO, val::LogRank)
#     show(io, ev.time)
#     iscensored(ev) && print(io, '+')
#     return nothing
# end

"""
    logrank(df, event, by)

Performs log-rank test for the groups `by`.
"""
function logrank(df, event, by)

    # number of groups
    ba = completecases(df[!,[event,by]])
    df2 = df[ba,:]
    groups = sort(unique(df2[!,by]))
    n_groups = length(groups)

    # perform Kaplan-Meier analysis
    km = Vector{KaplanMeier{Float64, Int64}}(undef,n_groups)
    times = zeros(Int64,n_groups)

    for (i,v) in 1:n_groups
        km[i] = fit(KaplanMeier, df2[ df2[!,by] .== v, event])
        times[i] = km[i].events.time[end]

    end

    # lengths
    ntimes = maximum(times)

    # events
    events = zeros(Int64, n_groups, ntimes)
    for i in 1:n_groups
        events[i, km[i].events.time] .= km[i].events.nevents
    end

    # N at risk
    atrisk = zeros(Int64, 2, ntimes)
    for i in 1:n_groups
        atrisk[i, 1] = km[i].events.natrisk[1]
        
        for j in 2:ntimes
            atrisk[i, j] = atrisk[i, j-1] - events[i, j-1]
        end
    end

    # Observed events
    O = vec(sum(events, dims=1))

    # Total N at risk
    N = vec(sum(atrisk, dims=1))

    # Observed Rate
    Or = O ./ N

    # Expected values
    E = zeros(Float64, n_groups, ntimes)
    for i in 1:n_groups
        E[i, :] = Or .* vec(atrisk[i, :])
    end

    o = vec(sum(events, dims=2))
    e = vec(sum(E, dims=2))

    chi2 = sum((o .- e) .^ 2 ./ e)

    dof = (length(o) - 1) * (length(e) - 1)
    pval = ccdf(Chisq(dof), chi2)

    return (o, e, size(df2,1), chi2, dof, pval)
end