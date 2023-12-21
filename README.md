# RefineGAN-unofficial

    Unofficial implemetation of the RefineGAN vocoder

----

The RefineGAN [https://arxiv.org/abs/2111.00962](https://arxiv.org/abs/2111.00962) vocoder sounds great, let's try to reproduce it!


### Quickstart

⚪ install

- `conda create -n refinegan & conda activate refinegan`
- install pytorch follow the [official guide](https://pytorch.org/get-started/locally/)
- `pip install requirements.txt`

⚪ inference

- `python infer.py`

⚪ train

- download the HiFiGAN repo provided [pretrained weights](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)
- `python train.py`


#### reference

Thanks to the code base & pretrained ckpt from official [HiFiGAN](https://github.com/jik876/hifi-gan) 🎉~

- thesis
  - MelGAN: [https://arxiv.org/abs/1910.06711](https://arxiv.org/abs/1910.06711)
  - HiFi-GAN: [https://arxiv.org/abs/2010.05646](https://arxiv.org/abs/2010.05646)
  - RefineGAN: [https://arxiv.org/abs/2111.00962](https://arxiv.org/abs/2111.00962)
- repo
  - HiFiGAN (official): [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)
    - pretrained weights: [https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)
  - MelGAN: [https://github.com/jaywalnut310/MelGAN-Pytorch](https://github.com/jaywalnut310/MelGAN-Pytorch)
  - RefineGAN (unofficial): [https://github.com/nikhilpinnaparaju/RefineGAN](https://github.com/nikhilpinnaparaju/RefineGAN)
  - RetuneGAN: [https://github.com/Kahsolt/TransTacoS-RetuneGAN](https://github.com/Kahsolt/TransTacoS-RetuneGAN)

----
by Armit
2023/11/26 
