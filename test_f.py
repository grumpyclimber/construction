# lets create a function for tweaking plots:
def spines(ax,yl='Price',xl='Gr Liv Area',title='No title!'):
    x1 = ax.spines['right'].set_visible(False)
    x2 = ax.spines['top'].set_visible(False)
    x3 = ax.spines['left'].set_linewidth(2)
    x4 = ax.spines['bottom'].set_linewidth(2)
    x5 = ax.set_ylabel(yl)
    x6 = ax.set_xlabel(xl)
    x7 = ax.set_title(title)
    return x1, x2, x3, x4, x5, x6
