The project will involve improving the prediction of multi-output drug sensitivities used in describe the drug response in cancer drug treatment. 

Such a project has ever been implemented: one followed the RandomForest while the other followed the Gaussian Process

We aim to adopt the multi-output Gaussian Process to achieve our goal. The areas of improvement include:
    - The use of sparse MOGP
        * linear model of coregionalisation
        * Variational inference
    - Custom kernels for genoic features (transferrable kernel)
    - Alternative variable selection methods to automatic relevance selection (KL-relevance and VAR selection)
    - Custom likelihood specifically genomic data
    - Feature engineering

At the end of the project, we will want to improve the performance and predictions of the multiple drug responses

## MOGP MODEL
This model will be implemented using GPy package developed by Machine Learning team in the University of Sheffield. Alternatively, it can also be implemented with GPflow package. Let's check both.

## EXPERIMENTS:
    - Implementing the multi-output Gaussian Process
        * Do we need to specific model for multi-output?
        * specific kernel and likelihood
    - Use the alternative variable selection methods on MOGP
        * VAR on MOGP
        * KL-Divergence on Sparse
    - Results:
        * Collect metrics on all these
    - Chech the prediction power.

    Experiment 1:
        * Is the sparse mogp predictions accurate
        * Test dataset
    Experiment 2:
        * VAR method effectively selecting features with predictive power
        * compare on MOGP and Sparse

    