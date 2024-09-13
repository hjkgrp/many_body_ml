# Many-body Expansion Based Machine Learning Models for Octahedral Transition Metal Complexes

 This repository contains all the data and code to reproduce the results and figures from the main paper. To install all utilities and dependencies run:

 ``
pip install -e .
 ``
 
 (tested on Python--3.8.19) 
 
 The individual steps can be executed in any order since all the intermediate data that is generated in each step already contained in this repository. However, to evalute/reprodce everything from scratch the scripts and notebooks have to be evaluated in the following order:

 1. The `/raw_data` folder contains a `.csv` of the target properties and the `.xyz` files for all four data sets. The notebook `/scripts_and_notebooks/preprocessing/featurize_raw_data.ipynb` evaluates the standard-RACs and ligand-RACs feature vectors and saves the results in four `.csv` files in the `/data` folder which are used to train the models in the following step.
 2. The two scripts in `/scripts_and_notebooks/training/` can be used to retrain the different KRR and NN models by changing the values of `model_type = ModelType.STANDARD_RACS` and `target = TargetProperty.SSE` in the main function call at the very bottom of the scripts. The resulting models are saved in the `/models` folder.
 3. The model performance scores in the tables of the manuscript are generated from the 3 notebooks in `/scripts_and_notebooks/scoring/`
 4. `/scripts_and_notebooks/plotting` contains notebooks to generate all figures in the main text and supporting material. The four main figures are generated in the following notebooks:  
    - Fig. 1: `parity_plot_sse_train_val_horizontal.ipynb` (output: `plots/parity_plot_sse_val_horizontal.pdf`)
    - Fig. 2: `interpolation_plot_sse_horizontal.ipynb` (output: `plots/interpolation_plot_sse_horizontal_krr_1.pdf`)
    - Fig. 3: `parity_plot_sse_ligand_test_horizontal.ipynb` (output: `plots/parity_plot_see_horizontal_lig_test_nn.pdf`)
    - Fig. 4: `interpolation_plot_orbitals.ipynb` (output: `plots/interpolation_plot_homo_ls_nn_1.pdf`)
    - Fig. 5: `feature_importance_nn_two_body.ipynb` (output: `plots/feature_importance_nn_two_body.pdf`)
