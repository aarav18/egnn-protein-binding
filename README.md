# **Predicting Protein-Ligand Complex Binding Affinities With Equivariant Graph Neural Networks**

### **Project Statement**

This project uses a distinctive form of Graph Neural Networks (GNNs) called Equivariant Graph Neural Networks (EGNNs). EGNNs are used to predict the binding affinity of protein-ligand pairs. Given the 3-dimensional molecular structure of a protein-ligand pair as input, the model outputs the predicted binding affinity of the molecule complex. The molecular structure is generally given in the Simplified Molecular Input Line Entry System (SMILES) format, which must be parsed and parameterized. The binding affinity is a single scalar value, usually in units of moles per liter.

### **Dataset**

The primary dataset used in this project is the PDB-Bind dataset, which was engineered specifically to assist in modeling protein-ligand interactions. PDB-Bind is a publicly available dataset which is a collection of experimentally measured binding affinities for the protein-ligand complexes in the Protein Data Bank (PDB). It is periodically updated to reflect recent developments in binding affinity data for various complexes.

The PDB-Bind dataset is available at: https://www.pdbbind-plus.org.cn/.

There are two primary subsets of the PDB-Bind dataset which are commonly used. The “Core” subset contains information about 193 different protein-ligand complexes. The much larger “Refined” subset contains information about 4852 protein-ligand complexes. For this project, the “Core” subset of the PDB-Bind dataset was used. This is chiefly due to the fact that the “Refined” subset consists of a massive amount of raw data, and training a complex deep learning model on this data requires a level of hardware, compute, and time that was not accessible for this project.

Fortunately, training a model on the “Core” dataset yields results sufficient to evaluate the generalization ability of the model, even if it does not hold up to state-of-the-art performance.
In terms of the data themselves, the primary features of importance are the 3-dimensional structures of the protein and ligand. These structures are represented in the Simplified Molecular Input Line Entry System (SMILES) format, which is fundamentally a string of various ASCII characters. This form is convenient for readability and interpretability, as it encodes 3-dimensional structure in a compact and readable format. However, deep learning models such as the noted Equivariant Graph Neural Network are unable to directly take in a SMILES string as input without preprocessing, often called featurization.

The featurization of molecules involves transforming molecular representations (in this case, a SMILES string) into a numerical vector which can be input into the machine learning model. There are various popular methods of featurization, each suited to different model architectures. This project makes use of a Graph-Based featurization technique, which treats molecules as graphs. In this methodology, atoms are nodes and bonds are edges. This is ideal for a model like an Equivariant Graph Neural Network. The resulting graph representation (after featurization) captures both the structure and the chemical properties of the protein-ligand complex, which is essential for accurately predicting the binding affinity of the pair.

The labels in the dataset are much simpler than the features. The label for each sample is the binding affinity of the given protein-ligand pair. This is represented as a single scalar value (usually in units of moles per liter), which frames the task as a regression problem.

### **Evaluation**

For evaluation, I used 5-fold cross-validation to evaluate the model’s performance on the validation dataset. In terms of the metrics used to evaluate the model, I used the Pearson R2 score (the coefficient of determination), which is common in regression problems like this one.
The general trend of training and validation loss is below.

![](https://drive.google.com/uc?export=view&id=1XGIuJuRU2dL9EZIdvV2piC950yynMSRd)

The trend shows a clear steady decrease in the training loss, but the validation loss stays quite constant over time. This suggests that the model was overfitting. This could potentially be addressed by introducing dropout layers in the Equivariant Graph Neural Network architecture, but this would take more experimentation.
Below are the final results:

![](https://drive.google.com/uc?export=view&id=12l6AaleF4XzrJ4Ej9dYiHiIPaEtQuHZW)

These final results show more clear signs of overfitting, as the Pearson correlation on the train dataset is much higher than the Pearson score on the validation and test datasets. Again, some methods to address overfitting could significantly improve the results. I believe that overfitting became such a problem because the more complicated the architecture of a deep learning model, the more parameters and biases it has, and the more likely the model is to overfit the data. Equivariant Graph Neural Networks are, by design, complex, so they are more likely to overfit the data and less likely to generalize to unseen protein-ligand complexes.

### **Further Work**

There are numerous avenues for further work on this project:
- Using Cylindrical or Spherical coordinates to encode distance rather than Cartesian coordinates
- Using Physics-Informed Graph Neural Networks with a notion of Equivariance
- Train on the full “Refined” subset of the PDB-Bind database and benchmark on CASF-2016
