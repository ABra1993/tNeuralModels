# tNeuralModels
The current repository contains the implementation of the Delayed Normalization (DN) model developed by Zhou et al. (2019). The model’s architecture is defined by a LNG structure (Linear, Nonlinear, Gain control), where as an input it receives a stimulus timecourse, and as output it produces a neuronal response.

## Files and directories
### tm_DN_simulate_model.py
This script simulates the model using on of two variations of the DN model:
- Zhou et al. : Described in the paper below (see References). This model contains 7 parameters.
- Groen et al. : Unpublished variation of the DN model. This model contains 9 parameters, where the impulse response function (IRF) for the linear computation has two additional parameters.

### tm_DN_fit_model.py
Fits the model using the least squares algorithm.

### models
contains two classes (i.e. Model_Zhou_et_al and Model_Groen_et_al) where the two variations of the DN model are implemented.

### utils
Contains the following scripts:
- visualization: function to visualize the output of the model simulation (toy example and with fitted parameters)
- functions : contains two response functions used for the (non)linear computation of the neural output.
- tm_objective : functions for fitting the model with data (least sqaures method)

## References
Zhou, J., Benson, N. C., Kay, K., & Winawer, J. (2019). Predicting neuronal dynamics with a delayed gain control model. PLoS computational biology, 15(11), e1007484.
