from GPy import kern, models
import numpy as np


def get_kernel(num_of_features, AddKern_loc, config, output_dim, Dim, noise_variance=1e-6, ARD=False):
    """
    Create a Gaussian process kernel with specified parameters.
    Args:
        input_dim (int): Dimensionality of the input.
        variance (float, optional): Variance of the kernel. Defaults to 1.0.
        lengthscale (float, optional): Lengthscale of the kernel. Defaults to 1.0.
        noise_variance (float, optional): Variance of the noise. Defaults to 1e-6.
    Returns:
        multi-output kernel.
    """
    if variance is None:
        variance = np.random.rand(num_of_features)
    else:
        variance = np.array(variance)
    
    if lengthscale is None:
        lengthscale = np.random.rand(num_of_features)
    else:
        lengthscale = np.array(lengthscale)

    
    for i in range(1, num_of_features):
        combined_kernel = combined_kernel * kern.RBF(AddKern_loc[i]-AddKern_loc[i-1],active_dims=list(np.arange(AddKern_loc[i-1], AddKern_loc[i]), lengthscale=lengthscale, noise_var=noise_variance))

    combined_kernel.rbf.lengthscale = float(config.lengthscale)* np.sqrt(Dim) * np.random.rand()
    combined_kernel.rbf.variance.fix()
    for i in range(1, num_of_features):
        eval("combined_kernel.rbf_"+str(i)+".lengthscale.setfield(float(lengthscale)* np.sqrt(Dim) * np.random.rand(), np.float64)")
        eval("combined_kernel.rbf_" + str(i) + ".variance.fix()")
    
    # coreginalisation matrix for a multi-output kernel
    B = kern.Coregionalize(1, output_dim=output_dim, rank=config.rank)
    multi_kernel = combined_kernel.prod(B)

    # multi_kernel.B.W.fix(0)
    # multi_kernel.B.kappa.fix(0)
    
    return multi_kernel


def create_and_initialize_model(Xtrain, Ytrain, kern, Ntasks, config, model_type='GP', num_inducing=None):
    """
    Create a Gaussian process model with specified parameters.
    Args:
        Xtrain (numpy array): Input data for training.
        Ytrain (numpy array): Output data for training.
        kern (multi-output kernel): a combination of kernel for each cancer feature.
        Ntasks (int): Number of tasks in the multi-output problem.
        config (Config): Configuration parameters for the model.
        model_type (str): Type of Gaussian process model, either 'sparse' or 'full'.
        num_inducing (int, optional): Number of inducing points. Defaults to None.
    Returns:
        Multi-Output Gaussian process model.
    """
    model = models.GPRegression(Xtrain, Ytrain, kern)

    Init_Weights = float(config.weight) * np.random.randn(Ntasks, config.rank)
    model.kern.coregion.W = Init_Weights

    if config.Train:
        model.optimize(optimizer='lbfgsb',max_iters=int(config.N_iter))
        # model.optimize()
    else:
        pass
        # Here we load the model bash*:
        # m_trained = str(config.bash)
        # print("loading model ", m_trained)
        # model[:] = np.load('./GDSC2_Codes_ANOVAFeatures/Best_Model_Drug' + config.drug_name + '_MelanomaGDSC2_GPy_ANOVA_ExactMOGP_ProdKern/m_' + m_trained + '.npy')
    
    return model