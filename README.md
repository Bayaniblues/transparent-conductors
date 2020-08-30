## Transparent conductors

Predicting transparent conductors with Support Vector machines. There are two targets, formation energy is the amount of energy lost when a gas turns into a solid. A low formation energy means that the molecule is more stable because of less entropy. The second target is the bandgap. The bandgap determines if the molecule will be a good conductor or insulator. We will make a model from the bandgap energy for simplicity.

For feature engineering, there is only 10 features that can be used for example the ratio of Al, In, Ga elements and lattice features. The other features, id and spacegroup were excluded to avoid leakage. The number_of_total_atoms can be excluded but at no change to the R2 score.

We made a density plot and found the MSE for the baseline.

Next we made a Random forest regressor for the tree model and a Support Vector machine for our polynomial model. A polynomial hyperperameter was chosen over the linear one because of it's better R2 score but the svm_data method can still create a linear model.

The Poly Support Vector Reggressor is reliable because of it's R2 score of 96 was better than the other methods.

The quantum computing library pennylane was also used to find the ground state, but currently the model has bad accuracy because 10 molecules is hard to test with the current state of quantum computing.


## To run notebook

    pip install -r requirementsdev.txt
    
    jupyter notebook
    
## To run app 

    pip install -r requirements.txt
    
    python app.py

    