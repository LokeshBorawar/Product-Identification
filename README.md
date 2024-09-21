# Product identification in a mall or a grocery store.

- To identify products in a store, this repository utilizes a CLIP model that generates feature embeddings for various items. By leveraging cosine similarity, the model compares these embeddings against a database of pre-saved features from the store inventory to accurately identify products. The implementation specifically employs several versions of CLIP, including CLIP-ViT-L-14, CLIP-ViT-B-16, and CLIP-ViT-B-32, for generating these product embeddings.

- However, prior preprocessing is essential. The process begins by detecting and identifying each person in the scene, assigning them a unique ID. Once individuals are identified, the next step is to capture images of the products they hold, as the model requires product images for recognition. This task can be challenging. Alternatively, the system can utilize cameras in shopping baskets or monitor items at checkout to obtain product images, thereby streamlining the identification process. For demonstration purposes, products in the frames captured by [CCTV](CCTV) are manually [cropped](Demo1/zoom).

- In [Demo 1](Demo1), inventory images are sourced from the [Foodi-ML](https://github.com/Glovo/foodi-ml-dataset.git) dataset, ensuring that each product image is of higher resolution than the zoomed CCTV images. In contrast, [Demo 2](Demo2) involves a similar technique for both the inventory images and the zoomed CCTV images, where both sets are zoomed in on the products, resulting in lower resolution. This approach highlights the differences in image quality and emphasizes the challenges faced in accurately identifying products from varying sources.

- Please see all results from evey CLIP model, below are just from CLIP-ViT-L-14.
- These are Demo1 results. The first shows which two of the best inventory images match the product. And, the second one shows the similarity matrix of the product to all inventory items.

  ![cid1](Demo1/clip-ViT-L-14_composite_image.png)

  ![cid1](Demo1/clip-ViT-L-14_confusion_matrix.png)

- These are Demo2 results.

  ![cid2](Demo2/clip-ViT-L-14_composite_image.png)

  ![cid2](Demo2/clip-ViT-L-14_confusion_matrix.png)
 

## Contacts:
- borawarlokesh26@gmail.com


## Credits:
- [foodi-ml-dataset](https://github.com/Glovo/foodi-ml-dataset.git)