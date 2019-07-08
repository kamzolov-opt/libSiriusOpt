def show(data, namefile, legends=[],  colors=[], title="", xlabel="", ylabel=""):
    import matplotlib.pyplot as plt

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    colors = iter(colors if colors else ["b", "g", "r", "go:"])

    plt.legend(tuple([plt.plot(el, next(colors)) for el in data]), tuple(legends), loc = 'best')

    plt.grid()
    plt.savefig(f'{namefile}.png', format = 'png')
