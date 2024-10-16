from dataset import DatasetID

import matplotlib.animation as animplt
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pickle

def plotAnimation(gradients, accumulated_gradients, figures_dir):
    # ===== Create Animation =====
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

    # Plot heatmap animation of gradient
    fig = plt.figure()
    gs = fig.add_gridspec(len(gradients[0]), 2)
    ax_grad = fig.add_subplot(gs[:, 0])
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
            ax = fig.add_subplot(gs[counter, 0])
            # layer-wise scaling between 0 and 1
            # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / grad_layer_scaling[counter]), vmin=0, vmax=1)
            # global scaling between 0 and 1
            ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / grad_scaling), vmin=0, vmax=1)
            ax.set_axis_off()

    ax_grad = fig.add_subplot(gs[:, 1])
    ax_grad.axis("off")
    ax_grad.set_title("Accumulated Gradient")

    accgrad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*accumulated_gradients)]
    accgrad_layer_scaling = [als if als != 0 else 1 for als in accgrad_layer_scaling]
    accgrad_scaling = np.max(np.array(accgrad_layer_scaling))
    def showAccGrad(i):
        for counter, layer_grad in enumerate(accumulated_gradients[i]):
            primfac = prime_factors(layer_grad.size)
            p = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 0]))
            q = int(np.prod([elem for idx, elem in enumerate(primfac) if idx % 2 == 1]))
            ax = fig.add_subplot(gs[counter, 1])
            # layer-wise scaling of gradient between 0 and 1
            # ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / accgrad_layer_scaling[counter]), vmin=0, vmax=1)
            # global scaling of gradient between 0 and 1
            ax.imshow(np.absolute(layer_grad.reshape((min(p, q), max(p, q))) / accgrad_scaling),
                vmin=0, vmax=1)
            ax.set_axis_off()

    def showGradients(i):
        fig.suptitle(f'Epoch {i}')
        showGrad(i)
        showAccGrad(i)
    anim = animplt.FuncAnimation(fig, showGradients, frames=len(gradients), interval=1000)
    anim.save(figures_dir/r'animation.gif', writer=animplt.PillowWriter(fps=1))


def plotHighestValue(gradients, accumulated_gradients, figures_dir):
    # compute factors to scale the gradients to an overall maximum of 1 for the heatmap (layerwise or global)
    grad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*gradients)]
    grad_layer_scaling = [gls if gls != 0 else 1 for gls in grad_layer_scaling]
    grad_scaling = np.max(np.array(grad_layer_scaling))

    accgrad_layer_scaling = [np.max(np.absolute(np.array(layer_grads).flatten()))
        for layer_grads in zip(*accumulated_gradients)]
    accgrad_layer_scaling = [als if als != 0 else 1 for als in accgrad_layer_scaling]
    accgrad_scaling = np.max(np.array(accgrad_layer_scaling))

    # ===== Obtain the highest gradient elements =====
    grad_layer_highest_elements = [[np.max(np.absolute(layer_grad.flatten())) / grad_layer_scaling[counter]
            for counter, layer_grad in enumerate(grad)]
        for grad in gradients]
    grad_highest_elements = np.max(np.array(grad_layer_highest_elements), axis=0)
    grad_layer_highest_elements = list(zip(*grad_layer_highest_elements))

    accgrad_layer_highest_elements = [[np.max(np.absolute(layer_grad.flatten())) / accgrad_layer_scaling[counter]
            for counter, layer_grad in enumerate(accgrad)]
        for accgrad in accumulated_gradients]
    accgrad_highest_elements = np.max(np.array(accgrad_layer_highest_elements), axis=0)
    accgrad_layer_highest_elements = list(zip(*accgrad_layer_highest_elements))

    plt.figure()
    for counter, glhe in enumerate(grad_layer_highest_elements):
        plt.plot(glhe, label=f'Grad Layer {counter}')
    for counter, alhe in enumerate(accgrad_layer_highest_elements):
        plt.plot(alhe, linestyle='dashed', label=f'Accgrad Layer {counter}')
    plt.title("Highest Gradient Elements")
    plt.legend(loc="upper center")
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
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

    # extract loss and additional metric
    statistics["loss"] = [metr[list(metr.keys())[0]] for metr in metrics]
    statistics["metric"] = [metr[list(metr.keys())[-1]] for metr in metrics]


    # ===== Plot statistics =====
    def scaleToMax1(arr):
        return np.array(arr) / np.max(np.array(arr))

    NUM_EPOCHS = len(gradients)

    plt.figure()

    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm"]), label="L1 Norm")
    # plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm"]), label="L2 Norm")
    # plt.plot(scaleToMax1(range(0, len(statistics["l1_norm_individual"])))*NUM_EPOCHS,
    #   scaleToMax1(statistics["l1_norm_individual"]), label="L1 Norm Individual")
    # plt.plot(scaleToMax1(range(0, len(statistics["l2_norm_individual"])))*NUM_EPOCHS,
    #   scaleToMax1(statistics["l2_norm_individual"]), label="L2 Norm Individual")
    # plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm_acc"]), label="L1 Norm Acc.")
    # plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm_acc"]), label="L2 Norm Acc.")

    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm_standardized"]), label="L1 Norm Std.")
    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm_standardized"]), label="L2 Norm Std.")
    # plt.plot(scaleToMax1(range(0, len(statistics["l1_norm_individual_standardized"])))*NUM_EPOCHS,
    #   scaleToMax1(statistics["l1_norm_individual_standardized"]), label="L1 Norm Individual Std.")
    # plt.plot(scaleToMax1(range(0, len(statistics["l2_norm_individual_standardized"])))*NUM_EPOCHS,
    #   scaleToMax1(statistics["l2_norm_individual_standardized"]), label="L2 Norm Individual Std.")
    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l1_norm_acc_standardized"]), label="L1 Norm Acc. Std.")
    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["l2_norm_acc_standardized"]), label="L2 Norm Acc. Std.")

    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["loss"]), label="Loss")
    plt.plot(range(1, NUM_EPOCHS+1), scaleToMax1(statistics["metric"]), label="Metric")

    plt.legend(loc="upper center")
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    plt.savefig(figures_dir/r'statistics.png')


def plotAll(dataset_id, gradients, individual_gradients, metrics, figures_dir):
    # Compute cumsum of gradients (i.e., accumulated gradient)
    accumulated_gradients = np.cumsum(gradients, axis=0)

    plotAnimation(gradients, accumulated_gradients, figures_dir)
    plotHighestValue(gradients, accumulated_gradients, figures_dir)
    plotStatistics(gradients, individual_gradients, accumulated_gradients, metrics, figures_dir)

def main():
    dataset_id = DatasetID.Mnist

    FIGURES_DIR = Path(f'figures_{dataset_id.name}')
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # load gradients from disk
    filehandler = open(FIGURES_DIR/r'gradients.pkl', "rb")
    gradients = pickle.load(filehandler)
    filehandler.close()
    # load individual gradients from disk
    filehandler = open(FIGURES_DIR/r'individual_gradients.pkl', "rb")
    individual_gradients = pickle.load(filehandler)
    filehandler.close()
    # load metrics from disk
    filehandler = open(FIGURES_DIR/r'metrics.pkl', "rb")
    metrics = pickle.load(filehandler)
    filehandler.close()

    plotAll(dataset_id, gradients, individual_gradients, metrics, FIGURES_DIR)

if __name__ == "__main__":
    main()
