
## Build Basic Generative Adversarial Networks 
### Week 1: Intro to GANs
| 内容 | 备注 |
| - | - | 
|[教学ppt](./Build_Basic_Generative_Adversarial_Networks/week1/C1_W1.pdf)| |
|[编程实验: 一个训练好的GAN模型的探索](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C1W1_(Colab)_Pre_trained_model_exploration.ipynb) | |
|[编程实验: 了解GAN的输入以及它对输出的影响](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C1W1_(Colab)_Inputs_to_a_pre_trained_GAN.ipynb) | |
|[编程作业: 第一个GAN](./Build_Basic_Generative_Adversarial_Networks/week1/C1W1_Your_First_GAN.ipynb)| | 

### Week 2: Deep Convolutional GANs
| 内容 | 备注 |
| - | - |
|[教学ppt](./Build_Basic_Generative_Adversarial_Networks/week2/C1_W2.pdf)| |
|[博客: 转置卷积以及交互式的展示棋盘格效应](https://distill.pub/2016/deconv-checkerboard/)| |
|[编程作业: DCGAN](./Build_Basic_Generative_Adversarial_Networks/week2/C1_W2_Assignment.ipynb) | |
|[论文: DCGAN](https://arxiv.org/abs/1511.06434) | |
|[编程实验: 利用TGAN生成视频](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C1W2_Video_Generation_(Optional).ipynb)| 在本笔记本中，您将从论文 [Temporal Generative Adversarial Nets with Singular Value Clipping](https://arxiv.org/abs/1611.06624) (Saito, Matsumoto, & Saito, 2017) 中了解 TGAN，以及它在图像生成中的起源|

### Week 3: Wasserstein GANs with Gradient Penalty
| 内容 | 备注 |
| - | - | 
|[教学ppt](./Build_Basic_Generative_Adversarial_Networks/week3/C1_W3.pdf)| |
|[编程作业: wgan_gp](./Build_Basic_Generative_Adversarial_Networks/week3/C1W3_WGAN_GP.ipynb)| |
|[编程实验: SN-GAN](./Build_Basic_Generative_Adversarial_Networks/week3/SNGAN.ipynb)| 在本笔记本中, 您将了解并实现spectral normalization，这是一种用于稳定鉴别器训练的权重归一化技术, 在[ Spectral Normalization for Generative Adversarial Networks ](https://arxiv.org/abs/1802.05957) (Miyato 等人 2018) 中提出. |
|[编程实验: GAN在生物信息学中的应用ProteinGAN](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/ProteinGAN.ipynb)| |
|[论文: Wasserstein GAN](https://arxiv.org/abs/1701.07875) | |
|[论文: Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) | |
|[博客: From GAN to WGAN](https://lilianweng.github.io/posts/2017-08-20-gan/)| |

### Week 4: Conditional GAN & Controllable Generation
| 内容 | 备注 |
| - | - | 
|[教学ppt](./Build_Basic_Generative_Adversarial_Networks/week4/C1_W4.pdf)| |
|[编程作业: Conditional GAN](./Build_Basic_Generative_Adversarial_Networks/week4/C1W4A_Build_a_Conditional_GAN.ipynb)| |
|[编程实验: InfoGAN](./Build_Basic_Generative_Adversarial_Networks/week4/InfoGAN.ipynb)| |
|[论文: Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) | |
|[作业: Controllable Generation](./Build_Basic_Generative_Adversarial_Networks/week4/C1W4B_Controllable_Generation.ipynb) | |
|[论文：Interpreting the Latent Space of GANs for Semantic Face Editing](https://arxiv.org/pdf/1907.10786.pdf)| |


## Build Better Generative Adversarial Networks
### Week 1: Evaluation of GANs
|内容|备注|
|	-	|	-	|
|[教学ppt](./Build_Better_Generative_Adversarial_Networks/week1/C2_W1.pdf)| |
|[论文: A Note on the Inception Score](https://arxiv.org/abs/1801.01973)|FID为什么超过Inception score, 本文解释了Inception score的缺点|
|[论文: HYPE: A Benchmark for Human eYe Perceptual Evaluation of Generative Models](https://arxiv.org/abs/1904.01121) |GAN的人工评估和HYPE(Human eYe Perceptual Evaluation)|
|[论文: Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991)|在GAN上使用准确率和召回率指标|
|[编程作业: Fréchet Inception Distance](./Build_Better_Generative_Adversarial_Networks/week1/C2W1_Assignment.ipynb) | |
|[编程实验: Perceptual Path Length](./Build_Better_Generative_Adversarial_Networks/week1/PPL.ipynb) | |
|[博客: FID 和 IS 回顾之Fréchet Inception Distance](https://nealjean.com/ml/frechet-inception-distance/) |FID和IS的回顾 |
|[博客: FID 和 IS 回顾之 How to measure GAN performance?](https://jonathan-hui.medium.com/gan-how-to-measure-gan-performance-64b988c47732) |FID和IS的回顾|


### Week 2: GAN Disadvantages and Bias
| 内容 | 备注 |
|-|-|
|[教学ppt](./Build_Better_Generative_Adversarial_Networks/week2/C2_W2.pdf)| |
|[编程实验: Alternatives: Variational Autoencoders (VAEs)](./Build_Better_Generative_Adversarial_Networks/week2/C2W2_VAE.ipynb) | |
|[编程实验: Score-based Generative Modeling](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_(Optional_Notebook)_Score_Based_Generative_Modeling.ipynb) |这是score-based的生成模型的简短指南, 这是一系列基于[estimating gradients of the data distribution](https://arxiv.org/abs/1907.05600)的方法. 他们在不需要对抗训练的情况下获得了与 GAN 相当的高质量样本, 并且被一些人认为是 GAN 的[新竞争者](https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/). |
|[博客: 机器学习中的偏见](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) | |
|[博客: Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf) [Machine Learning Glossary: Fairness](https://developers.google.com/machine-learning/glossary/fairness)| 公平的定义 |
|[论文: A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/abs/1908.09635)| |
|[博客: Does Object Recognition Work for Everyone](https://arxiv.org/abs/1906.02659) [What a machine learning tool that turns Obama white can (and can't) tell us about AI bias](https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias) | 如何发现现有材料（模型、数据集、框架等）中的偏见以及如何防止它 |
|[编程作业: Bias](./Build_Better_Generative_Adversarial_Networks/week2/C2W2_Assignment.ipynb) | |
|[编程实验: GAN Debiasing](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_GAN_Debiasing_(Optional).ipynb) |了解通过潜在空间去偏进行公平属性分类, [Fair Attribute Classification through Latent Space De-biasing](https://princetonvisualai.github.io/gan-debiasing/)|
|[NeRF: Neural Radiance Fields](https://colab.research.google.com/drive/18DladhUz7_U8iBkkQxMBk2f7C2NAvPCC?usp=sharing) |学习如何使用神经辐射场仅使用几个输入视图生成复杂 3D 场景的新视图，最初由 NeRF 提出：[Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al. 2020 ）。尽管 2D GAN 在高分辨率图像合成方面取得了成功，但 NeRF 已迅速成为实现高分辨率 3D 感知 GAN 的流行技术。 |

### Week 3: StyleGAN and Advancements

| 内容 | 备注 |
| - | - |
|[教学ppt](./Build_Better_Generative_Adversarial_Networks/week3/C2_W3.pdf) | |
|[编程作业: Components of StyleGAN](./Build_Better_Generative_Adversarial_Networks/week3/C2W3_Assignment.ipynb) | |
|[编程实验: Components of StyleGAN2](./Build_Better_Generative_Adversarial_Networks/week3/) | |
|[编程实验: BigGAN](./Build_Better_Generative_Adversarial_Networks/week3/BigGAN.ipynb) | |
|[论文: StyleGAN](https://arxiv.org/abs/1812.04948) | |
|[博客: StyleGAN的另外一种解释](https://jonathan-hui.medium.com/gan-stylegan-stylegan2-479bdf256299) | |
|[编程实验: Finetuning GAN](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W3_FreezeD_(Optional).ipynb) |了解并实现 [Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs (Mo et al. 2020)](https://arxiv.org/abs/2002.10964) 中提出的微调方法，该方法介绍了冻结鉴别器上层的概念在微调。具体来说，将微调预训练的 StyleGAN 以从人脸生成动漫面孔 |


## Apply Generative Adversarial Networks

### Week 1: GANs for Data Augmentation and Privacy
| 内容 | 备注 |
| - | - |
|[教学ppt](./Apply_Generative_Adversarial_Networks/week1/C3_W1.pdf) | |
|[论文: Automated Data Augmentation](https://arxiv.org/abs/1909.13719) | 混合数据增强技术和自动增强策略感 |
|[编程作业: Data Augmentation](./Apply_Generative_Adversarial_Networks/week1/C3W1_Assignment.ipynb)| |
|[编程实验: Generative Teaching Networks](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W1_Generative_Teaching_Networks_(Optional).ipynb) |在本笔记本中, 您将实现 Generative Teaching Network (GTN), 该网络首次在[Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data](https://arxiv.org/abs/1912.07768)中介绍: 通过学习生成合成训练数据来加速神经架构搜索（Such et al. 2019）. 本质上, GTN 由生成合成数据的生成器（即教师）和针对某些任务接受此数据训练的学生组成. GTN 和 GAN 之间的主要区别在于 GTN 模型是协同工作的（而不是对抗性的）. |
|[论文: Talking Heads](https://arxiv.org/abs/1905.08233)| 如何使用 GAN 来创建会说话的头像和 deepfakes |
|[论文: De-identification](https://arxiv.org/abs/1902.04202)| 如何在保留基本面部属性以隐藏身份的同时对面部进行去识别（匿名化）的信息 |
|[论文: GAN Fingerprints](https://arxiv.org/abs/1811.08180)| 担心区分真实图像和伪造的 GAN 生成图像? 看看 GAN 是如何留下指纹的!|

### Week 2: Image-to-Image Translation with Pix2Pix
| 内容 | 备注 |
| - | - |
|[教学ppt](./Apply_Generative_Adversarial_Networks/week2/C3_W2.pdf)| |
|[编程作业: U-Net](./Apply_Generative_Adversarial_Networks/week2/C3W2A_Assignment.ipynb) | |
|[编程作业: Pix2Pix](./Apply_Generative_Adversarial_Networks/week2/C3W2B_Assignment.ipynb) | |
|[论文: Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)| |
|[编程实验: Pix2PixHD](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W2_Pix2PixHD_(Optional).ipynb) | 在本笔记本中，您将了解 Pix2PixHD，它从语义标签映射中合成高分辨率图像。 Pix2PixHD 在[ High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585)中提出，通过多尺度架构、改进的对抗性损失和实例映射对 Pix2Pix 进行了改进。 |
|[编程实验: Super-resolution GAN](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W2_SRGAN_(Optional).ipynb) | 在本笔记本中，您将了解 Super-Resolution GAN (SRGAN)，这是一种将图像分辨率提高 4 倍的 GAN，在[ Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)(Ledig et al. 2017) 中提出。您还将实现架构和训练代码，并能够在 CIFAR 数据集上对其进行训练。|
|[论文: PatchGAN](https://arxiv.org/abs/1803.07422) | GAN 如何填充图像的裁剪部分？了解 PGGAN 如何使用 PatchGAN 做到这一点!|
|[编程实验: GauGAN](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W2_GauGAN_(Optional).ipynb) |在本笔记本中，您将了解 GauGAN，它从您实现和训练的语义标签映射中合成高分辨率图像。 GauGAN 基于[ Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)（Park 等人，2019 年）中提出的一种特殊的反归一化技术 |


### Week 3: Unpaired Translation with CycleGAN
| 内容 | 备注 |
| - | - |
|[教学ppt](./Apply_Generative_Adversarial_Networks/week3/C3_W3.pdf)| |
|[编程作业: CycleGAN](./Apply_Generative_Adversarial_Networks/week3/C3W3_Assignment.ipynb) | |
|[论文: CycleGAN](https://arxiv.org/abs/1703.10593) | |
|[阅读材料: CycleGAN for Medical Imaging](https://www.nature.com/articles/s41598-019-52737-x.pdf) | |
|[编程实验: MUNIT](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C3W3_MUNIT_(Optional).ipynb) | 在本笔记本中，您将了解并实现 MUNIT，这是一种无监督图像到图像转换的方法，在`Multimodal Unsupervised Image-to-Image Translation (Huang et al. 2018)` 中提出。 |


