# DGCMIA
Dual-Granularity Cross-Modal Identity Association for Weakly-Supervised Text-to-Person Image Matching (ACMMM2025))

## Pipeline

![7](C:\Users\yongl\Desktop\CodeRepository\DGCMIA\7.png)

## Usage

### **Environment Setup**

we use single RTX4090 24G GPU for training and evaluation. 

```python
conda env create -f environment.yml
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

### Prepare Datasets

Download the CUHK-PEDES dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description), ICFG-PEDES dataset from [here](https://github.com/zifyloo/SSAN) and RSTPReid dataset form [here](https://github.com/NjtechCVLab/RSTPReid-Dataset).

Organize them in `your dataset root dir` folder as follows:

```
|-- your dataset root dir/
|   |-- <CUHK-PEDES>/
|       |-- imgs
|            |-- cam_a
|            |-- cam_b
|            |-- ...
|       |-- reid_raw.json
|
|   |-- <ICFG-PEDES>/
|       |-- imgs
|            |-- test
|            |-- train 
|       |-- ICFG_PEDES.json
|
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

### Prepare Backbone

Download the [ViT-B/16 weight](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt), and move it to the ./model/clip_pretrain/ folder.

## Training

```python
python train.py \
--gpu_id 0 \
--batch_size 64 \
--dataset_name "CUHK-PEDES"  \
--loss_names  "itc+itc1+cons+memory_soft" \    
--num_epoch 40  
```

## Testing

```python
python test.py --config_file 'path/to/model_dir/configs.yaml'
```

## Model Weights

[Model & log for CUHK-PEDES](https://drive.google.com/file/d/16Xjblb3FgThDGLRt1i3DqFeog15mXTN6/view?usp=sharing)

[Model & log for ICFG-PEDES](https://drive.google.com/file/d/1LzZJ2EUEL4hlULDuVrntu2ore8itpbne/view?usp=sharing)

[Model & log for RSTPReid](https://drive.google.com/file/d/1w983bPk7NmRn5vYTH8WrUF5ZAIjXbb2m/view?usp=sharing)

## Acknowledgments

Some components of this code implementation are adopted from [IRRA](https://github.com/anosorae/IRRA). We sincerely appreciate for their contributions.

## Contact

If you have any questions, please feel free to contact me (sylmail99@163.com)






