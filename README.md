# A Copula-infused Graph Neural Network for Cell Type Classification in Single-cell RNA Sequencing Data

Cell type identification using single-cell RNA sequencing (scRNA-seq) is critical for understanding disease mechanisms, improving disease diagnosis, and advancing drug discovery. This process involves classifying scRNA-seq data into clusters of single cells. While deep learning approaches have made significant strides in enhancing clustering performance in high-dimensional scRNA-seq data compared to traditional clustering methods, there is still a need for more comprehensive, extendable end-to-end frameworks that integrate advanced statistical clustering with deep learning approaches.

To address this issue, we proposed the Copula-infused Graph Neural Network(scCopulaGNN) which combines the Guassian copula and the deep learning method Graph Convolutional Network(GCN).We used the processed cell-cell graph matrix with the algorithm, applying on four different public scRNA-seq datasets and simulated datasets. In this way, we show that scCopulaGNN is generally competent on a variety of datasets, while possessing strengths in classification accuracy, robustness to sparsity, and scalability compared with existing methods. These results highlight scCopulaGNN potential as an effective tool for cell type classification in single-cell transcriptomics, providing more elaborate details about cellular diversity and function.

Key words: single-cell RNA sequencing, cell type classification, graph neural networks, copula theory

We acknowledge Ma et al. for their source code repository https://github.com/jiaqima/CopulaGNN we used in the study.

