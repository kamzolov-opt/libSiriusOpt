def show(data, namefile, labels=[], title="", xlabel="", ylabel="", dpi=800):
    import matplotlib.pyplot as plt

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    while len(labels) < len(data):
        labels.append("1")
    labels = iter(labels)
    #   plt.legend(tuple([plt.plot(el) for el in data]), tuple(legends), loc = 'best')
    for el in data:
        plt.plot(el, label=next(labels))
    plt.legend(loc="upper right")

    plt.grid()
    plt.savefig(f'{namefile}.png', format='png', dpi=dpi)
