import matplotlib.pyplot as plt


def plot_all_signals(signals, labels):
    label_set = set(labels)
    fig, axs = plt.subplots(len(set(labels)))
    fig.suptitle('Vertically stacked subplots')
    for i, l in enumerate(label_set):
        sigs = [v for v, label in zip(signals, labels) if label == l]
        for s in sigs:
            plot_signal(s, axs[i])
    plt.show()


def plot_signal(signal, plot):
    plot.plot([n for n in range(len(signal))], signal)


def plot_clusters(values, labels, title=""):
    for l in set(labels):
        y = [v for v, label in zip(values, labels) if label == l]
        plt.scatter([l for i in range(len(y))], y, alpha=0.5)
        plt.title(title)
