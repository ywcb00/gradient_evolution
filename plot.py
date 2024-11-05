from dataset import DatasetID

import matplotlib.animation as animplt
from matplotlib.legend_handler import HandlerTuple
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle

def plotAnimation(gradients, accumulated_gradients, figures_dir):
    NUM_EPOCHS = len(gradients)

    # ===== Create Animation =====
    # Function to obtain the prime factors for gradient layer reshaping
    def prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    # Plot heatmap animation of gradients
    fig = plt.figure(figsize=(10, 8), dpi=300)
    fig_gridspec = fig.add_gridspec(1, 3)

    # Gradients
    grad_gridspec = fig_gridspec[0, 0].subgridspec(len(gradients[0]), 1)
    ax_grad = fig.add_subplot(grad_gridspec[:, 0])
    ax_grad.axis("off")
    ax_grad.set_title("Gradient")

    # compute factors to scale the gradients to an overall maximum of 1 for the heatmap (layerwise or global)
    grad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*gradients)]
    grad_layer_scaling = [gls if gls != 0 else 1 for gls in grad_layer_scaling]
    grad_scaling = np.max(np.array(grad_layer_scaling))
    def showGrad(i):
        for counter, layer_grad in enumerate(gradients[i]):
            primfac = prime_factors(layer_grad.size)
            p = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 0]))
            q = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 1]))
            ax = fig.add_subplot(grad_gridspec[counter, 0])
            # layer-wise scaling between 0 and 1
            ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / grad_layer_scaling[counter]), vmin=0, vmax=1)
            # global scaling between 0 and 1
            # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / grad_scaling), vmin=0, vmax=1)
            ax.set_axis_off()

    # Dissolving Gradients
    accumulated_absolute_gradients = np.cumsum(np.absolute(gradients), axis=0)
    dissolving_gradients = accumulated_absolute_gradients - np.absolute(accumulated_gradients)

    dissgrad_gridspec = fig_gridspec[0, 1].subgridspec(len(dissolving_gradients[0]), 1)
    ax_dissgrad = fig.add_subplot(dissgrad_gridspec[:, 0])
    ax_dissgrad.axis("off")
    ax_dissgrad.set_title("Dissolving Gradient")

    dissgrad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*dissolving_gradients)]
    dissgrad_layer_scaling = [dls if dls != 0 else 1 for dls in dissgrad_layer_scaling]
    dissgrad_scaling = np.max(np.array(dissgrad_layer_scaling))
    def showDissGrad(i):
        for counter, layer_grad in enumerate(dissolving_gradients[i]):
            primfac = prime_factors(layer_grad.size)
            p = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 0]))
            q = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 1]))
            ax = fig.add_subplot(dissgrad_gridspec[counter, 0])
            # layer-wise scaling of gradient between 0 and 1
            ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / dissgrad_layer_scaling[counter]), vmin=0, vmax=1)
            # global scaling of gradient between 0 and 1
            # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / dissgrad_scaling),
            #     vmin=0, vmax=1)
            ax.set_axis_off()

    # Accumulated Gradients
    accgrad_gridspec = fig_gridspec[0, 2].subgridspec(len(accumulated_gradients[0]), 1)
    ax_accgrad = fig.add_subplot(accgrad_gridspec[:, 0])
    ax_accgrad.axis("off")
    ax_accgrad.set_title("Accumulated Gradient")

    accgrad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*accumulated_gradients)]
    accgrad_layer_scaling = [als if als != 0 else 1 for als in accgrad_layer_scaling]
    accgrad_scaling = np.max(np.array(accgrad_layer_scaling))
    def showAccGrad(i):
        for counter, layer_grad in enumerate(accumulated_gradients[i]):
            primfac = prime_factors(layer_grad.size)
            p = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 0]))
            q = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 1]))
            ax = fig.add_subplot(accgrad_gridspec[counter, 0])
            # layer-wise scaling of gradient between 0 and 1
            ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / accgrad_layer_scaling[counter]), vmin=0, vmax=1)
            # global scaling of gradient between 0 and 1
            # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / accgrad_scaling),
            #     vmin=0, vmax=1)
            ax.set_axis_off()

    def showGradients(i):
        fig.suptitle(f'Gradient Animation - Epoch {i}')
        showGrad(i)
        showDissGrad(i)
        showAccGrad(i)

    # create colorbar
    cbar = fig.colorbar(None, ax=fig.get_axes())
    fig.subplots_adjust(right=0.75)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['low', 'medium', 'high'])

    anim = animplt.FuncAnimation(fig, showGradients, frames=NUM_EPOCHS, interval=1000)
    anim.save(figures_dir/r'animation.gif', writer=animplt.PillowWriter(fps=1))


def plotHighestValue(gradients, accumulated_gradients, figures_dir):
    NUM_EPOCHS = len(gradients)

    # compute factors to scale the gradients to an overall maximum of 1 for the heatmap (layerwise or global)
    # grad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
    #     for layer_grads in zip(*gradients)]
    # grad_layer_scaling = [gls if gls != 0 else 1 for gls in grad_layer_scaling]
    # grad_scaling = np.max(np.array(grad_layer_scaling))

    accgrad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*accumulated_gradients)]
    accgrad_layer_scaling = [als if als != 0 else 1 for als in accgrad_layer_scaling]
    accgrad_scaling = np.max(np.array(accgrad_layer_scaling))

    # ===== Obtain the highest gradient elements =====
    grad_layer_highest_elements = [[np.max(np.absolute(layer_grad.flatten())) / accgrad_layer_scaling[counter]
            for counter, layer_grad in enumerate(grad)]
        for grad in gradients]
    grad_highest_elements = np.max(np.array(grad_layer_highest_elements), axis=0)
    grad_layer_highest_elements = np.array(list(zip(*grad_layer_highest_elements)))

    accgrad_layer_highest_elements = [[np.max(np.absolute(layer_grad.flatten())) / accgrad_layer_scaling[counter]
            for counter, layer_grad in enumerate(accgrad)]
        for accgrad in accumulated_gradients]
    accgrad_highest_elements = np.max(np.array(accgrad_layer_highest_elements), axis=0)
    accgrad_layer_highest_elements = np.array(list(zip(*accgrad_layer_highest_elements)))

    plt.figure(figsize=(10, 8), dpi=300)
    glhe_lines = list()
    for counter, glhe in enumerate(grad_layer_highest_elements):
        glhe_lines.append(plt.plot(range(1, NUM_EPOCHS+1), glhe, color=f'C{counter}',
            label=f'Grad Layer {counter}')[0])
    alhe_lines = list()
    for counter, alhe in enumerate(accgrad_layer_highest_elements):
        alhe_lines.append(plt.plot(range(1, NUM_EPOCHS+1), alhe, color=f'C{counter}',
            linestyle='dashed', label=f'Accgrad Layer {counter}')[0])

    glhe_lines = glhe_lines[:10]
    alhe_lines = alhe_lines[:10]

    plt.title("Highest Gradient Elements")
    plt.xlabel("# Epoch")
    if(NUM_EPOCHS <= 20):
        plt.xticks(np.arange(1, NUM_EPOCHS+1, 1))
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.legend([tuple(glhe_lines), tuple(alhe_lines)], ["Gradient Layers", "Accumulated Gradient Layers"],
        numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, borderpad=0,
        handlelength=len(glhe_lines)*0.75, loc="center right")
    plt.savefig(figures_dir/r'highestgradelement.png')

def plotStatistics(gradients, individual_gradients, accumulated_gradients, metrics, figures_dir):
    # ===== Compute statistics =====
    statistics = dict()

    # compute the norm of the gradients as the sum of the layer gradient norms
    statistics["l1_norm"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1)
        for layer_grad in grad]) for grad in gradients]
    statistics["l1_norm_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) / layer_grad.size
        for layer_grad in grad]) for grad in gradients]
    statistics["l2_norm"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2)
        for layer_grad in grad]) for grad in gradients]
    statistics["l2_norm_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) / layer_grad.size
        for layer_grad in grad]) for grad in gradients]

    statistics["acc_l1_norm"] = np.cumsum(statistics["l1_norm"])
    statistics["acc_l2_norm"] = np.cumsum(statistics["l2_norm"])
    statistics["acc_l1_norm_standardized"] = np.cumsum(statistics["l1_norm_standardized"])
    statistics["acc_l2_norm_standardized"] = np.cumsum(statistics["l2_norm_standardized"])

    # compute the norm of the individual gradients as the sum of the layer gradient norms
    statistics["l1_norm_individual"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1)
        for layer_grad in grad]) for grad in individual_gradients]
    statistics["l1_norm_individual_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) / layer_grad.size
        for layer_grad in grad]) for grad in individual_gradients]
    statistics["l2_norm_individual"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2)
        for layer_grad in grad]) for grad in individual_gradients]
    statistics["l2_norm_individual_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) / layer_grad.size
        for layer_grad in grad]) for grad in individual_gradients]

    # compute the norm of the accumulated gradients (cumsum) as the sum of the layer gradient norms
    statistics["l1_norm_acc"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1)
        for layer_grad in accgrad]) for accgrad in accumulated_gradients]
    statistics["l1_norm_acc_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=1) / layer_grad.size
        for layer_grad in accgrad]) for accgrad in accumulated_gradients]
    statistics["l2_norm_acc"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2)
        for layer_grad in accgrad]) for accgrad in accumulated_gradients]
    statistics["l2_norm_acc_standardized"] = [np.sum([np.linalg.norm(layer_grad.flatten(), ord=2) / layer_grad.size
        for layer_grad in accgrad]) for accgrad in accumulated_gradients]

    # compute "dissolving norm" as a metric for the disappearing/equalizing information of gradients during training
    statistics["l1_dissolving_norm"] = statistics["acc_l1_norm"] - statistics["l1_norm_acc"]
    statistics["l1_dissolving_norm_standardized"] = statistics["acc_l1_norm_standardized"] - statistics["l1_norm_acc_standardized"]
    statistics["l2_dissolving_norm"] = statistics["acc_l2_norm"] - statistics["l2_norm_acc"]
    statistics["l2_dissolving_norm_standardized"] = statistics["acc_l2_norm_standardized"] - statistics["l2_norm_acc_standardized"]

    # extract loss and additional metric
    statistics["metrics"] = {metr_name: [metr[metr_name] for metr in metrics] for metr_name in metrics[0].keys()}
    statistics["metrics"].pop('loss')

    # ===== Scale data for plotting =====
    def scaleToMax1(arr):
        return np.array(arr) / np.max(np.array(arr))

    # plot_data = dict()

    # plot_data["l1_norm"] = scaleToMax1(statistics["l1_norm"])
    # plot_data["l1_norm_standardized"] = scaleToMax1(statistics["l1_norm_standardized"])
    # plot_data["l2_norm"] = scaleToMax1(statistics["l2_norm"])
    # plot_data["l2_norm_standardized"] = scaleToMax1(statistics["l2_norm_standardized"])
    # plot_data["l1_norm_individual"] = scaleToMax1(statistics["l1_norm_individual"])
    # plot_data["l1_norm_individual_standardized"] = scaleToMax1(statistics["l1_norm_individual_standardized"])
    # plot_data["l2_norm_individual"] = scaleToMax1(statistics["l2_norm_individual"])
    # plot_data["l2_norm_individual_standardized"] = scaleToMax1(statistics["l2_norm_individual_standardized"])
    # plot_data["l1_norm_acc"], plot_data["acc_l1_norm"] = scaleToMax1(
    #     [statistics["l1_norm_acc"], statistics["acc_l1_norm"]])
    # plot_data["l1_norm_acc_standardized"], plot_data["acc_l1_norm_standardized"] = scaleToMax1(
    #     [statistics["l1_norm_acc_standardized"], statistics["acc_l1_norm_standardized"]])
    # plot_data["l2_norm_acc"], plot_data["acc_l2_norm"] = scaleToMax1(
    #     [statistics["l2_norm_acc"], statistics["acc_l2_norm"]])
    # plot_data["l2_norm_acc_standardized"], plot_data["acc_l2_norm_standardized"] = scaleToMax1(
    #     [statistics["l2_norm_acc_standardized"], statistics["acc_l2_norm_standardized"]])
    # plot_data["l1_dissolving_norm"] = scaleToMax1(statistics["l1_dissolving_norm"])
    # plot_data["l1_dissolving_norm_standardized"] = scaleToMax1(statistics["l1_dissolving_norm_standardized"])
    # plot_data["l2_dissolving_norm"] = scaleToMax1(statistics["l2_dissolving_norm"])
    # plot_data["l2_dissolving_norm_standardized"] = scaleToMax1(statistics["l2_dissolving_norm_standardized"])

    # plot_data["loss"] = scaleToMax1(statistics["loss"])
    # plot_data["metric"] = scaleToMax1(statistics["metric"])

    # ===== Plot statistics =====
    NUM_EPOCHS = len(gradients)

    # plt.figure(figsize=(10, 8), dpi=300)

    # L1 Norms
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l1_norm"], label="L1 Norm")
    # plt.plot(scaleToMax1(range(0, len(statistics["l1_norm_individual"])))*NUM_EPOCHS,
    #   plot_data["l1_norm_individual"], label="L1 Norm Individual")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l1_norm_acc"], label="L1 Norm Acc.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l1_dissolving_norm"], label="L1 Diss. Norm")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["acc_l1_norm"], label="L1 Norm CumSum")

    # L1 Norms Standardized
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l1_norm_standardized"], label="L1 Norm Std.")
    # plt.plot(scaleToMax1(range(0, len(plot_data["l1_norm_individual_standardized"])))*NUM_EPOCHS,
    #     plot_data["l1_norm_individual_standardized"], alpha=0.5, label="L1 Norm Individual Std.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l1_norm_acc_standardized"], label="L1 Norm Acc. Std.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l1_dissolving_norm_standardized"], label="L1 Diss. Norm Std.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["acc_l1_norm_standardized"], label="L1 Norm Std. CumSum")

    # L2 Norms
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l2_norm"], label="L2 Norm")
    # plt.plot(scaleToMax1(range(0, len(statistics["l2_norm_individual"])))*NUM_EPOCHS,
    #   plot_data["l2_norm_individual"], label="L2 Norm Individual")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l2_norm_acc"], label="L2 Norm Acc.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l2_dissolving_norm"], label="L2 Diss. Norm")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["acc_l2_norm"], label="L2 Norm CumSum")

    # L2 Norms Standardized
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l2_norm_standardized"], label="L2 Norm Std.")
    # plt.plot(scaleToMax1(range(0, len(statistics["l2_norm_individual_standardized"])))*NUM_EPOCHS,
    #   plot_data["l2_norm_individual_standardized"], alpha=0.5, label="L2 Norm Individual Std.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l2_norm_acc_standardized"], label="L2 Norm Acc. Std.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["l2_dissolving_norm_standardized"], label="L2 Diss. Norm Std.")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["acc_l2_norm_standardized"], label="L2 Norm Std. CumSum")

    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["loss"], label="Loss")
    # plt.plot(range(1, NUM_EPOCHS+1), plot_data["metric"], label="Metric")

    # plt.legend(loc="upper center")
    # ax = plt.gca()
    # ax.get_yaxis().set_visible(False)
    # plt.savefig(figures_dir/r'statistics.png')

    plt.figure(figsize=(10, 8), dpi=300)
    plt.plot(range(1, NUM_EPOCHS+1), statistics["l1_norm_standardized"],
        linestyle="-", marker="|", label="Norm of Gradient")
    plt.plot(range(1, NUM_EPOCHS+1), statistics["l1_norm_acc_standardized"],
        linestyle="-", marker="|", label="Norm of Accumulated Gradient")
    plt.plot(range(1, NUM_EPOCHS+1), statistics["acc_l1_norm_standardized"],
        linestyle="-", marker="|", label="Accumulated Norm of Gradient")
    plt.vlines(range(1, NUM_EPOCHS+1), statistics["l1_norm_acc_standardized"], statistics["acc_l1_norm_standardized"],
        linestyle="dashed", color="gray", label="Dissolving Gradient Norm")
    plt.title("Gradient Norm")
    plt.xlabel("# Epoch")
    if(NUM_EPOCHS <= 20):
        plt.xticks(np.arange(1, NUM_EPOCHS+1, 1))
    plt.ylabel("L1 Norm")
    plt.legend(loc="upper left")
    plt.savefig(figures_dir/r'gradient_norms.png')

    if individual_gradients:
        plt.figure(figsize=(10, 8), dpi=300)
        plt.plot(scaleToMax1(range(0, len(statistics["l1_norm_individual_standardized"])))*NUM_EPOCHS,
            statistics["l1_norm_individual_standardized"], linestyle="-", label="L1 Norm Individual Std.")
        plt.title("Individual Gradient Norm")
        plt.xlabel("# Epoch")
        plt.xticks(np.arange(1, NUM_EPOCHS+1, 1))
        plt.ylabel("L1 Norm")
        plt.legend(loc="upper right")
        plt.savefig(figures_dir/r'individual_gradients.png')

    plt.figure(figsize=(10, 8), dpi=300)
    for metr_name, metr_arr in statistics["metrics"].items():
        plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(metr_arr), linestyle="-", marker="|", label=metr_name)
    plt.title("Loss and Metrics")
    plt.xlabel("# Epoch")
    if(NUM_EPOCHS <= 20):
        plt.xticks(np.arange(1, NUM_EPOCHS+1, 1))
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.legend(loc="upper center")
    plt.savefig(figures_dir/r'loss.png')


def plotAll(dataset_id, gradients, individual_gradients, metrics, figures_dir):
    # Compute cumsum of gradients (i.e., accumulated gradient)
    accumulated_gradients = np.cumsum(gradients, axis=0)

    plotAnimation(gradients, accumulated_gradients, figures_dir)
    plotHighestValue(gradients, accumulated_gradients, figures_dir)
    plotStatistics(gradients, individual_gradients, accumulated_gradients, metrics, figures_dir)

def main():
    dataset_id = DatasetID.FordA

    FIGURES_DIR = Path(f'figures_{dataset_id.name}')
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # load gradients from disk
    filehandler = open(FIGURES_DIR/r'gradients.pkl', "rb")
    gradients = pickle.load(filehandler)
    filehandler.close()
    try:
        # load individual gradients from disk
        filehandler = open(FIGURES_DIR/r'individual_gradients.pkl', "rb")
        individual_gradients = pickle.load(filehandler)
        filehandler.close()
    except:
        individual_gradients = []
    # load metrics from disk
    filehandler = open(FIGURES_DIR/r'metrics.pkl', "rb")
    metrics = pickle.load(filehandler)
    filehandler.close()

    plotAll(dataset_id, gradients, individual_gradients, metrics, FIGURES_DIR)

if __name__ == "__main__":
    main()
