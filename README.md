# BiGAMi
Implementation of "Bi-Objective Genetic Algorithm Fitness Function for Feature Selection on Microbiome Datasets".


## Abstract

The relationship between the host and the microbiome, or the assemblage of microorganisms (including bacteria, archaea, fungi, and viruses), has been proven crucial for its health and disease development. The high dimensionality of microbiome datasets has often been addressed as a major difficulty for data analysis, such as the use of Machine Learning (ML) and Deep Learning (DL) models. Here we present BiGAMi, a bi-objective genetic algorithm fitness function for feature selection in microbial datasets to train high-performing phenotype classifiers. The proposed fitness function allowed us to build classifiers that outperformed the baseline performance estimated by the original studies by using as few as 0.04% to 2.32% features of the original dataset. In 19 out of 21 classification exercises, BiGAMi achieved its results by selecting 6-68% fewer features than the highest performance of a Sequential Forward Feature Selection algorithm. This study showed that the application of a bi-objective GA fitness function against microbiome datasets succeeded in selecting small subsets of bacteria whose contribution to understood diseases and the host state was already experimentally proven. Applying this feature selection approach to novel diseases is expected to quickly reveal the microbes most relevant to a specific condition.


## Getting started

The following steps will guide you to start the search for the most predictive features on your dataset of interest.

1. Clone the BiGAMi code repository:
    ```
    git clone https://github.com/mikeleske/BiGAMi.git
    ```
   
2. Update the `path` variable in `src/config.py` to point to your dataset storage:
    ```
    'path': r'/path/to/folder',
    ```

3. (Optional) Modify config parameters in `src/config.py` as required.
   
4. (Optional) Store your dataset in the `input` folder.

5. (Optional) Create a classification task for your own dataset in `src/tasks.py`.

6. Execute the GA feature selection search:
    ```
    python src/main_ga.py
    ```