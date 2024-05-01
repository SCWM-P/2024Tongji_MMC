import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def set_figure(
        fig, title, content,
        cmap='gray'
):
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    im = ax.imshow(content, cmap=cmap)
    plt.colorbar(im, ax=ax)
    return fig

