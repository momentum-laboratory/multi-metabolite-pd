import plotly
import numpy as np
from cmcrameri import cm

def plotly_cm(cm_choice):
    # Convert to Plotly colorscale
    positions = np.linspace(0, 1, len(cm_choice.colors))
    plotly_cm = [
        [float(pos), f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})']
        for pos, (r, g, b) in zip(positions, cm_choice.colors)
    ]
    plotly_cm[0][1] = '#000000'

    return plotly_cm

# Create custom Viridis colormap with black for 0 values
custom_viridis = np.array(plotly.colors.sequential.Viridis)
custom_viridis[0] = '#000000'  # Set black for 0 values

custom_magma = np.array(plotly.colors.sequential.Magma)
custom_magma[0] = '#000000'  # Set black for 0 values

custom_hot = plotly.colors.sequential.Hot
custom_hot[0] = '#000000'  # Set black for 0 values

custom_plotly3 = np.array(plotly.colors.sequential.Inferno)
custom_plotly3[0] = '#000000'  # Set black for 0 values

custom_aggrnyl = np.array(plotly.colors.sequential.Aggrnyl)
custom_aggrnyl[0] = '#000000'  # Set black for 0 values

custom_magma = np.array(plotly.colors.sequential.Magma)
custom_magma[0] = '#000000'  # Set black for 0 values

custom_cividis = np.array(plotly.colors.sequential.Cividis)
custom_cividis[0] = '#000000'  # Set black for 0 values

custom_plasma = np.array(plotly.colors.sequential.Plasma)
custom_plasma[0] = '#000000'  # Set black for 0 values

custom_jet = np.array(plotly.colors.sequential.Jet)
custom_jet[0] = '#000000'  # Set black for 0 values

custom_greysr = plotly.colors.sequential.Greys_r
custom_greysr[0] = '#000000'  # Set black for 0 values

custom_brnr = plotly.colors.sequential.Brwnyl_r
custom_brnr[0] = '#000000'  # Set black for 0 values

custom_tealr = plotly.colors.sequential.Teal_r
custom_tealr[0] = '#000000'  # Set black for 0 values

plotly_lipari = plotly_cm(cm.lipari)

plotly_navia = plotly_cm(cm.navia)

# Manually add separate colorbars
def colorbarformatter(color, cmin, cmax, skip, format='reg', diagonal_ticks=False, hide_first_label=False):
    # Generate tick values
    tickvals = list(np.arange(cmin, cmax + skip, skip))  # Increment by skip

    # Generate tick text and handle the percentage format
    if format == '%':
        ticktext = [f'{i:.0f}%' if i == int(i) else f'{i:.1f}%' for i in tickvals]  # Shortest format
    else:
        ticktext = [f'{i:.0f}' if i == int(i) else f'{i:.1f}' for i in tickvals]  # Shortest format

    # Hide the first label if requested
    if hide_first_label:
        ticktext[0] = ''  # Empty string for the first tick label

    # Build the colorbar dictionary
    colorbar = {
        'colorscale': color,
        'cmin': cmin,
        'cmax': cmax,
        'colorbar': {
            'tickvals': tickvals,
            'ticktext': ticktext,
        }
    }

    # Add diagonal tick option
    if diagonal_ticks:
        colorbar['colorbar']['tickangle'] = 45  # Diagonal ticks

    return colorbar

color_bar_dict = {
    'mask': colorbarformatter(plotly.colors.sequential.Greys_r, 0, 1, 0.5),
    'colorbar_highres': {'colorscale': plotly.colors.sequential.Greys_r},
    'colorbar_t1w': colorbarformatter(plotly_lipari, 1000, 2000, 500),
    'colorbar_t2w': colorbarformatter(plotly_navia, 30, 70, 20),
    'colorbar_fs_mt': colorbarformatter(custom_viridis, 5, 15, 5),  # , format='%',
    'colorbar_ksw_mt': colorbarformatter(custom_magma, 0, 75, 25),
    'colorbar_fs_noe': colorbarformatter(custom_viridis, 0.4, 2, 0.4),
    'colorbar_ksw_noe': colorbarformatter(custom_magma, 0, 25, 5),
    'colorbar_fs_amide': colorbarformatter(custom_viridis, 0.2, 0.8, 0.2),  # , format='%'
    'colorbar_ksw_amide': colorbarformatter(custom_magma, 0, 100, 50),
    'colorbar_fs': colorbarformatter(custom_viridis, 0, 25, 5),
    'colorbar_ksw': colorbarformatter(custom_magma, 6000, 7000, 500),
}

