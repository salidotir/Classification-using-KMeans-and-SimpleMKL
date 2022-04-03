# Classification-using-KMeans-and-SimpleMKL
Very large scale classification based on K-Means clustering &amp; Multi-Kernel SVM(SimpleMKL)

Here, we are going to implement the method proposed in this article, "[Very large scale classification based on K-Means clustering &amp; Multi-Kernel SVM(SimpleMKL)](https://dl.acm.org/doi/abs/10.1007/s00500-018-3041-0)" at ACM Digital Library.

## Modules:

The code has below modules:
- KMeans Clustering
  - Select nearest & furthest points of each cluster
- Duplicate Removal
  - Remove all duplicate data
- Outlier Detection
  - Remove the last ROT-data based on their outlier score
  - Method proposed in this article, "[Robust, Scalable Anomaly Detection for Large Collections of Images](https://ieeexplore.ieee.org/document/6693467)".
- Human Labeling
  - Do labeling for the new representative dataset
- SimpleMKL
  - Multi Kernel SVM
  - Method proposed in this article, "[Simplemkl](https://www.researchgate.net/publication/29623253_Simplemkl)".

## Datasets:

The method is run on two diffrent types of datasets, large scale & very large scale satasets.

The **large scale datasets** are:
* [Breast-W](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)) (683*10)
* [Messidor](https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set) (1151*20)
* [Car](https://archive.ics.uci.edu/ml/datasets/car+evaluation) (1728*6)
    - uacc vs other
* [Spambase](https://archive.ics.uci.edu/ml/datasets/spambase) (4601*57)

The **very large scale datasets** are:
* [Coil2000](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)) (9’822*85)
* [Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) (45’211*17)
* [Skin Segmentation](https://archive.ics.uci.edu/ml/datasets/skin+segmentation#) (245’057*4)
* [Covertype](https://archive.ics.uci.edu/ml/datasets/covertype) (581’012*54)
    - Aspen vs other

## Results:

Results can be seen at the end of [presentation file](https://github.com/salidotir/Classification-using-KMeans-and-SimpleMKL/blob/main/Presentation.pdf) uploaded in this repository.
