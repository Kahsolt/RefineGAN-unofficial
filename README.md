# RefineGAN-unofficial

    Unofficial implemetation of the RefineGAN vocoder

----

The [RefineGAN](https://arxiv.org/abs/2111.00962) vocoder sounds great, let's try to reproduce it!  
ℹ it seems to be highly inspired by HiFiGAN, UnivNet & UNet 🎉  


### Quickstart

⚪ install

- `conda create -n refinegan & conda activate refinegan`
- install pytorch follow the [official guide](https://pytorch.org/get-started/locally/)
- `pip install requirements.txt`

⚪ inference

- `python infer.py pretrained/UNIVERSAL_V1/g_02500000`
- `python infer.py pretrained/LJ_V3/generator_v3`

⚪ train (refine)

- download the HiFiGAN repo provided [pretrained checkpoints](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)
- `python train.py -c configs/config_v1.refine.json --load pretrained/UNIVERSAL_V1 --log_path log/test`


#### reference

Thanks to the code base & pretrained ckpt from official [HiFiGAN](https://github.com/jik876/hifi-gan) 🎉~

- thesis
  - MelGAN: [https://arxiv.org/abs/1910.06711](https://arxiv.org/abs/1910.06711)
  - HiFi-GAN: [https://arxiv.org/abs/2010.05646](https://arxiv.org/abs/2010.05646)
  - RefineGAN: [https://arxiv.org/abs/2111.00962](https://arxiv.org/abs/2111.00962)
- repo
  - HiFiGAN (official): [https://github.com/jik876/hifi-gan](https://github.com/jik876/hifi-gan)
    - pretrained: [https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing](https://drive.google.com/drive/folders/1-eEYTB5Av9jNql0WGBlRoi-WH2J7bp5Y?usp=sharing)
  - UnivNet (unofficial): [https://github.com/maum-ai/univnet](https://github.com/maum-ai/univnet)
    - pretrained: [univ_c16_0292](https://drive.google.com/file/d/1Iqw9T0rRklLsg-6aayNk6NlsLVHfuftv/view) / [univ_c32_0288.pt](https://drive.google.com/file/d/1QZFprpvYEhLWCDF90gSl6Dpn0gonS_Rv/view)
  - MelGAN: [https://github.com/jaywalnut310/MelGAN-Pytorch](https://github.com/jaywalnut310/MelGAN-Pytorch)
  - RefineGAN (unofficial): [https://github.com/nikhilpinnaparaju/RefineGAN](https://github.com/nikhilpinnaparaju/RefineGAN)
  - RetuneGAN: [https://github.com/Kahsolt/TransTacoS-RetuneGAN](https://github.com/Kahsolt/TransTacoS-RetuneGAN)

----
by Armit
2023/11/26 
