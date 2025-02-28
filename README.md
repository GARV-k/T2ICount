# T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting
## [Paper (ArXiv)](https://arxiv.org/abs/) 


![teaser](asset/teaser_CLIP_Count.png)

Official Implementation for CVPR 2025 paper T2ICount: Enhancing Cross-modal Understanding for Zero-Shot Counting.

## Preparation

**Environment:** Create a virtural environment use Anaconda, and install all dependencies.
```
conda create -n clipcount python=3.8 -y;
conda activate clipcount;
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch
# For GPU with CUDA version 11.x, please use:
# conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
**Data:** We conduct experiments over three datasets, you can download and use whichever you would like to test.
The three dataset could be downloaded at: [FSC-147](https://github.com/cvlab-stonybrook/LearningToCountEverything) | [CARPK](https://lafi.github.io/LPN/).
Notice that you have to download the annoations of FSC-147 separately from [their repo](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master/data).

Extract and put the downloaded data in the `data/` dir. The complete file structure should look like this. You don't have to download all the dataset for evaluation, but you must have FSC-147 if you want to train the model.
```
data
├─CARPK/
│  ├─Annotations/
│  ├─Images/
│  ├─ImageSets/
│
├─FSC/    
│  ├─gt_density_map_adaptive_384_VarV2/
│  ├─images_384_VarV2/
│  ├─FSC_147/
│  │  ├─ImageClasses_FSC147.txt
│  │  ├─Train_Test_Val_FSC_147.json
│  │  ├─ annotation_FSC147_384.json
│  
├─ShanghaiTech/
│  ├─part_A/
│  ├─part_B/
```

## Run the Code
**Train**. 

---
**Evaluation**. 
```

We provide a [pre-trained ckpt](https://drive.google.com/file/d/17Dj0tjd29lPGOGYEF5IrE8aPClXUjTrR/view?usp=drive_link) of our full model, which has similar quantitative result as presented in the paper. 

```

## Citation
Consider cite us if you find our paper is useful in your research :).
```
@article{jiang2023clip,
  title={CLIP-Count: Towards Text-Guided Zero-Shot Object Counting},
  author={Jiang, Ruixiang and Liu, Lingbo and Chen, Changwen},
  journal={arXiv preprint arXiv:2305.07304},
  year={2023}
}

```
