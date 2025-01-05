# Sg-CoSOD

Single-group Generalized RGB and RGB-D Co-Salient Object Detection
---
This paper has been online published by IEEE Transactions on Circuits and Systems for Video Technology.

[Paper link](https://ieeexplore.ieee.org/abstract/document/10789239)  DOI: 10.1109/TCSVT.2024.3514872

Abstract
---
Co-salient object detection (CoSOD) aims to segment the co-occurring salient objects in a given group of relevant images. Existing methods typically rely on extensive group training data to enhance the model's CoSOD capabilities. However, fitting prior knowledge of the extensive group results in a significant performance gap between the seen and out-of-sample image groups. Relaxing such a fitting with fewer prior groups may improve the generalization ability of CoSOD while alleviating the annotation burdens.  Hence, it is essential to explore the use of fewer groups during the training phase, such as using only single group, to pursue a highly generalized CoSOD model. We term this new setting as Sg-CoSOD, which aims to train a model using only a single group and effectively apply it to any unseen RGB and RGB-D CoSOD test groups. Towards Sg-CoSOD, it is important to ensure detection performance with limited data and release class dependency with only a single-group. Thus, we present a method, i.e., cross-excitation between saliency and 'Co', which decouples the CoSOD task into two parallel branches: 'Co' To Saliency (CTS) and Saliency To 'Co' (STC). The CTS branch focuses on mining group consensus to guide image co-saliency predictions, while the STC branch is dedicated to using saliency priors to motivate group consensus mining. Furthermore, we propose a Class-Agnostic Triplet (CAT) loss to constrain intra-group consensus while suppressing the model from acquiring class prior knowledge. Extensive experiments on RGB and RGB-D CoSOD tasks with multiple unknown groups show that our model has higher generalization capabilities (e.g., for large-scale datasets CoSOD3k and CoSal1k with multiple generalized groups, we obtain a gain of over 15% in F_m). Further experimental analyses also reveal that the proposed Sg-CoSOD paradigm has significant potential and promising prospects.

Network Architecture
====
![image](https://github.com/user-attachments/assets/f1ea943c-a2ec-499e-9029-e96359c245b8)

Quantitative results
===
![image](https://github.com/user-attachments/assets/3035a82e-85e4-4aac-b580-922989c53ed2)


Citation
===
```
@article{wang2024single,
  title={Single-group Generalized RGB and RGB-D Co-Salient Object Detection},
  author={Wang, Jie and Yu, Nana and Zhang, Zihao and Han, Yahong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```
