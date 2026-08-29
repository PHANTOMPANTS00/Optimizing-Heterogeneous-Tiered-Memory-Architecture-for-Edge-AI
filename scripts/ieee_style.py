import matplotlib.pyplot as plt
import seaborn as sns

def set_ieee_style(fig_type="single"):
    """
    Applies IEEE-compliant styling for plots.
    fig_type: "single" for 1-column (3.5 inch), "double" for 2-column (7.16 inch).
    Returns figsize tuple.
    """
    # Use seaborn whitegrid base but clean it up for academic style
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.9)
    
    # Fonts
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
    
    # Grids & Ticks
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.6
    plt.rcParams['grid.color'] = '#CCCCCC'
    
    # Borders
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = 1.0
    
    # Axes Formatter (Scientific Notation for IEEE)
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['axes.formatter.limits'] = (-3, 3)
    
    # Ticks
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    
    # Legend
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['legend.title_fontsize'] = 8
    
    # Size
    if fig_type == "single":
        figsize = (3.5, 2.5)
    elif fig_type == "double":
        figsize = (7.16, 2.8)
    else:
        figsize = (3.5, 2.5)
        
    return figsize

def get_hatch_patterns():
    """Returns a list of clear hatch patterns for bar charts."""
    return ['////', '\\\\\\\\', 'xxxx', '...', '++++', '----']

def get_markers():
    """Returns a list of distinct markers for line plots."""
    return ['o', 's', '^', 'D', 'v', 'x', '+']
