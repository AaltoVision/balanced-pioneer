PIONEER is a generative neural network model that learns  a well-structured representation of certain kinds of images, such as faces.

It can be used to modify your input images in various smart ways as in figures below, without losing sharpness in the output. The best-known generative models, GANs, cannot normally make this kind of general modifications to *existing input images* (unless they are extended with an encoder).

This paper marks a jump in resolution, quality and preservation of identity in face images over the previous PIONEER incarnation, and makes the feature modification capability more explicit. While the results were demonstrated on face data, the method is general.

<img src="samples/fig_manip/100.jpg" alt="alt text" width="15%" height="15%" style="border:5px solid black"/>
<img src="samples/fig_manip/101.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/102.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/103.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/104.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/105.jpg" alt="alt text" width="15%" height="15%"/>
<br/>
<img src="samples/fig_manip/200.jpg" alt="alt text" width="15%" height="15%" style="border:5px solid black"/>
<img src="samples/fig_manip/201.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/202.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/203.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/204.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/205.jpg" alt="alt text" width="15%" height="15%"/>
<br/>
<img src="samples/fig_manip/300.jpg" alt="alt text" width="15%" height="15%" style="border:5px solid black"/>
<img src="samples/fig_manip/301.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/302.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/303.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/304.jpg" alt="alt text" width="15%" height="15%"/>
<img src="samples/fig_manip/305.jpg" alt="alt text" width="15%" height="15%"/>


Figure: For real input images (left), our model can change various features (reconstruct, smile on/off, switch sex, rotate, add sunglasses) that it has learnt in a fully unsupervised manner - no class information was used during training.


## Abstract

We build on recent advances in progressively growing generative autoencoder models. These models can encode and reconstruct existing images, and generate novel ones, at resolutions comparable to Generative Adversarial Networks (GANs), while consisting only of a single encoder and decoder network. The ability to reconstruct and arbitrarily modify existing samples such as images separates autoencoder models from GANs, but the output quality of image autoencoders has remained inferior. The recently proposed PIONEER autoencoder can reconstruct faces in the 256x256 CelebAHQ dataset, but like IntroVAE, another recent method, it often loses the identity of the person in the process. We propose an improved and simplified version of PIONEER and show significantly improved quality and preservation of the face identity in CelebAHQ, both visually and quantitatively. We also show evidence of state-of-the-art disentanglement of the latent space of the model, both quantitatively and via realistic image feature manipulations. On the LSUN Bedrooms dataset, our model also improves the results of the original PIONEER. Overall, our results indicate that the PIONEER networks provide a way to photorealistic face manipulation.

## Materials

[Paper pre-print](https://arxiv.org/abs/1904.06145)

Video show-casing how to gradually apply various transformations on new input images:

<iframe width="515" height="290" src="https://www.youtube.com/embed/XhxKLkFVgjY" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Code (PyTorch): to be released later
Pre-trained models: to be released later

## Support

For all correspondence, please contact ari.heljakka@aalto.fi.

## Referencing

Please cite our work as follows:

```
@article{Heljakka+Solin+Kannala:2019,
      title = {Towards Photographic Image Manipulation with Balanced Growing of Generative Autoencoders},
     author = {Heljakka,Ari and Solin, Arno
               and Kannala, Juho},
    journal = {arXiv preprint arXiv:1904.06145},
       year = {2019}
}
```
