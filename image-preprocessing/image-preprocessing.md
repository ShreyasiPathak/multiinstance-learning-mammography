### Image Preprocessing

The image pre-processing algorithm to remove irrelevant information and excess background is described in the main paper. Figure below shows three examples of pre-processed images. It can be seen that the burned-in annotation with image types and the extra black background are removed in our pre-processed images. We generated 2 versions of the preprocesed images - 8 bit image and images with the original bit depth that was used to capture the image (found in the dicom metadata). This resulted in \mgm 12 bit images, \cbisddsm and VinDr as 16 bit images, in addition to the 8 bit images for all dataset.

Our reason for 2 versions of the dataset was due to the reproducibility experiments. For [1,2] models, we passed our 8 bit preprocessed images as input, whereas for [3], we passed our maximum bit depth preprocessed images as input, i.e. 16 bit for CBIS, VinDr and 12 bit for MGM. The reason for this difference is [3] explicitly stated that they used 16 bit images, whereas, the other two papers did not, motivating us to choose the standard image bit depth used for input to neural networks. 

| ![view1.png](LCC-image-preprocessing.png) | 
|:--:| 
| *LCC* |
| ![view2.png](LMLO-image-preprocessing.png) | 
| *LMLO* |
| ![view3.png](RMLO-imagepreprocessing.png) | 
| *RMLO* |

### References
1. E.-K. Kim, H.-E. Kim, K. Han, B. J. Kang, Y.-M. Sohn, O. H. Woo, and C. W. Lee, “Applying data-driven imaging biomarker in mammography for breast cancer screening: preliminary study,” Scientific reports, vol. 8, no. 1, pp. 1–8, 2018. <br/>
2. X. Shu, L. Zhang, Z. Wang, Q. Lv, and Z. Yi, “Deep neural networks with region-based pooling structures for mammographic image classification,” IEEE transactions on medical imaging, vol. 39, no. 6, pp. 2246–2255, 2020. <br/>
3. Y. Shen, N. Wu, J. Phang, J. Park, K. Liu, S. Tyagi, L. Heacock, S. G. Kim, L. Moy, K. Cho et al., “An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization,” Medical image analysis, vol. 68, p. 101908, 2021. <br/>
