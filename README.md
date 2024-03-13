# Collagen fiber synthesis and extraction networks

PyTorch implementation of the **collagen fiber centerline extraction network** proposed in\
[**Collagen Fiber Centerline Tracking in Fibrotic Tissue via Deep Neural Networks with Variational Autoencoder-based Synthetic Training Data Generation**](https://www.sciencedirect.com/science/article/pii/S1361841523002219),\
Hyojoon Park*, Bin Li*, Yuming Liu, Michael S. Nelson, Helen M. Wilson, Eftychios Sifakis, Kevin W. Eliceiri, \
Medical Image Analysis 2023.

![figure](/etc/figures/pipeline.png)

## Related repositories
* [Collagen fiber extraction and analysis in cancer tissue microenvironment](https://github.com/uw-loci/collagen-fiber-metrics)

 <div align="left">
  <img src="https://github.com/uw-loci/collagen-fiber-metrics/raw/master/thumbnails/output.png" width="800px" />
</div>

* [DuoVAE](https://github.com/hjoonpark/duovae)

<div align="left">
  <img src="https://github.com/hjoonpark/DuoVAE/blob/main/etc/figures/duovae_all_loop.gif" width="400px" />
</div>

---

## Installation

## Train

Command format is `python train.py <stage-number> --model-dir <model-directory>`, for example

### Stage I.
**DuoVAE** for generating collagen centerlines with desired centerline properties:

    python train.py 1

### Stage II.
**cGAN** for generating collagen images from collagen centerlines:

    python train.py 2

### Stage III.

**UNet** for extracting collagen centerlines from collagen images:

    python train.py 3

The outputs will be saved in the directories `output/stage1`, `output/stage2`, and `output/stage3`.

### Optional
To resume from a saved checkpoint, pass in `--model-dir` argument to a directory where the saved model (`.pt` files) is located and (optionally) set the number of starting epoch, for example

    # resume from saved model in "output/stage1/model" at epoch 1000
    python train.py 1 --model-dir output/stage1/model --starting-epoch 1000

## Results

### Stage 1 and 2

![figure](/etc/figures/result_stage1_stage2.png)

The figure shows representative outputs of the property-controlled data generation of DuoVAE on the collagen fiber dataset. The outputs are grouped into 6 panels according to the fiber properties: 

- (a) orientation (from left-oriented to right-oriented), 
- (b) alignment (from well-aligned to randomly organized), 
- (c) density (from sparse to dense), 
- (d) waviness (from straight to wavy), 
- (e) average length (from short to long), and 
- (f) length variation (from uniform lengths to random lengths). 

The first rows in each panel are the fiber centerlines generated by **DuoVAE**, and the second rows are the corresponding collage image generated by the **cGAN**. The generated centerlines exhibit the different values of fiber properties specified in the generating process.

### Stage 3

![figure](/etc/figures/result_stage3.png)

Representative results of input images captured using different objective magnifications (20× and 40×) are shown in the figure. The results show that the centerlines produced by the centerline extraction networks are consistently more similar to the ground truth annotations.

## Citation

    @article{park2023collagen,
             title={Collagen fiber centerline tracking in fibrotic tissue via deep neural networks with variational autoencoder-based synthetic training data generation},
             author={Park, Hyojoon and Li, Bin and Liu, Yuming and Nelson, Michael S and Wilson, Helen M and Sifakis, Eftychios and Eliceiri, Kevin W},
             journal={Medical Image Analysis},
             volume={90},
             pages={102961},
             year={2023},
             publisher={Elsevier}
    }
