Beyond Black and White: Adapting Models to Visual Domain Shift

## Project Overview

This project explores **Unsupervised Domain Adaptation (UDA)** techniques to address **covariate shift** in neural networks. We demonstrate how models trained on a source domain (grayscale MNIST digits) fail to generalize to a visually complex target domain (colored and textured MNIST-M digits), even when the underlying classification task is identical.

The project is implemented across three progressive stages:

1. **Baseline:** Demonstrating the impact of domain shift on model performance.
2. **Distance-based Alignment:** Using **Maximum Mean Discrepancy (MMD)** to align latent feature distributions.
3. **Adversarial Training (DANN):** Implementing **Domain-Adversarial Neural Networks** with a Gradient Reversal Layer (GRL) for superior feature alignment.

---

## The Problem: Covariate Shift

Neural networks typically assume that training and deployment data follow the same distribution. In this project, we utilize:

* **Source Domain :** MNIST (grayscale digit images).
* **Target Domain :** MNIST-M (digits blended with random color patches from the BSDS500 dataset).

While the label space  remains the same, the visual shift causes a standard CNN to struggle with learning dataset-invariant features.

---

## Methodology & Results

### Task 1: Baseline (Domain Shift Demonstration)

A standard CNN was trained exclusively on the source domain (MNIST). While it achieved high accuracy on the source test set, performance plummeted by **over 50%** when evaluated on the unseen MNIST-M target domain, proving the model's inability to generalize under covariate shift.

### Task 2: Distance-based Domain Adaptation

We implemented a "tug-of-war" optimization using **Energy Distance** (mathematically equivalent to MMD with a Gaussian kernel).

* **Approach:** Forced the network to output latent representations that are statistically identical for both domains.
* **Results:** Performance on MNIST-M jumped from **~43% to ~63%**.
* **Key Finding:** Larger batch sizes (up to 1024) significantly improved statistical distribution estimates, peaking at **79.36%** accuracy with tuned learning rates.

### Task 3: Adversarial Domain Adaptation (DANN)

Inspired by GANs, we implemented a **Domain-Adversarial Neural Network (DANN)**.

* **Architecture:** Modified the CNN to include a **Domain Discriminator** and a **Gradient Reversal Layer (GRL)**.
* **Mechanism:** The feature extractor actively learns to "confuse" the discriminator, ensuring features are discriminative for classification but uninformative regarding the domain.
* **Results:** This method achieved the highest robustness, with a mean accuracy of **89.04%** using a fixed alpha strategy.

### Results summary

![](Final_results.png)

--
## Requirements

To run these notebooks, you will need the following libraries:

* `torch` & `torchvision`
* `geomloss` (for MMD/Energy Distance computation)
* `numpy`
* `matplotlib`
* `tqdm`

## 💡 Key Technical Concepts

* **Gradient Reversal Layer (GRL):** Acts as an identity function during the forward pass but multiplies gradients by  during backpropagation to facilitate adversarial training.
* **Unsupervised Adaptation:** No target labels were ever used during the training process; the model adapts purely through feature alignment.
