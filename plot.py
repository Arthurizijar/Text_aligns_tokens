import matplotlib.pyplot as plt
import seaborn as sns

def draw_multibar_figure(x_list, y_list, label_list, path, share_x=False):
    
    sns.set_theme(style="whitegrid")
    f, axs = plt.subplots(len(x_list), 1, figsize=(9*len(x_list)+1, 8), sharex=False)
    if len(x_list) == 1:
        axs = [axs]
    for i, (x, y, label) in enumerate(zip(x_list, y_list, label_list)):
        if share_x and i != 0:
            sorted_combined = sorted(zip(y, x), key=lambda x: x[0], reverse=True)
            y = [item[0] for item in sorted_combined]
            x = [item[1] for item in sorted_combined]
        sns.barplot(x=x, y=y, hue=x, palette="coolwarm", ax=axs[i])
        axs[i].axhline(0, color="k", clip_on=False)
        axs[i].set_ylabel(label)

    leg = plt.legend()
    axs[-1].get_legend().remove()
    plt.savefig(path, bbox_inches='tight', dpi=300)


def draw_figure(data, path):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=range(len(data)), y=data)
    plt.xlabel('$i$ (Dismention)')
    plt.ylabel('$v_i$ (Variation)')
    plt.title('Contribution to the aligned tokens')
    plt.savefig(path, bbox_inches='tight', dpi=300)