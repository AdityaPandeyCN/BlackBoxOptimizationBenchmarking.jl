@recipe function plot_bbob(func::BBOBFunction{F, 2, M}; nlevels = 15, zoom = 1) where {F, M}

    markersize := 3
    titlefontsize := 10

    mx = 5.0f0
    
    if zoom > 1
        x = LinRange(func.x_opt[1] - 5f0/zoom, func.x_opt[1] + 5f0/zoom, 1000)
        y = LinRange(func.x_opt[2] - 5f0/zoom, func.x_opt[2] + 5f0/zoom, 1000)
    else
        x = LinRange(-mx, mx, 1000)
        y = LinRange(-mx, mx, 1000)
    end
    xlim = (minimum(x), maximum(x))
    ylim = (minimum(y), maximum(y))

    z = [func(SVector(xi, yi)) - func.f_opt for yi in y, xi in x]
    levels = 10 .^ (LinRange(-6, log10(maximum(z)), nlevels))

    @series begin
        seriestype := :heatmap
        
        xlabel := "x"
        ylabel := "y"
        cmap := :coolwarm
        alpha := 0.25
        aspectratio := 1
        title := string(func.f)
    
        x, y, z
    end

    @series begin
        seriestype := :contour
        levels := levels
        x, y, z
    end

    @series begin
        seriestype := :scatter
        x = [func.x_opt[1]]
        y = [func.x_opt[2]]
        c := "red"
        label := false
        xlim := xlim
        ylim := ylim

        x, y
    end

end