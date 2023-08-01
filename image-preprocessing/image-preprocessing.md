### Image Preprocessing

We first converted the dicom to png format. We generated the png images with the original bit depth that was used to capture the image (found in the dicom metadata). This resulted in MGM 12 bit images, CBIS and VinDr as 16 bit images. Then, we preprocessed the images to remove irrelevant information and excess background (the algorithm is described in the main paper and the code is shared in the repository). Figure below shows three examples of pre-processed images. It can be seen that the burned-in annotation with image types and the extra black background are removed in our pre-processed images. 

| ![view1.png](LCC-image-preprocessing.png) | 
|:--:| 
| *LCC* |
| ![view2.png](LMLO-image-preprocessing.png) | 
| *LMLO* |
| ![view3.png](RMLO-image-preprocessing.png) | 
| *RMLO* |
