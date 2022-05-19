# Calibration Matters: Tackling Maximization Bias in Large-scale Advertising Recommendation Systems

VAD-dlrm.ipynb contains code for DLRM model training, inference, and applying VAD on DLRM's model predictions. This file corresponds to experiments in Section 6.2, real-world data.

VAD-logistic-regression.ipynb contains code for Synthetic data experiments (Section 6.1).

Folder DLRM contains DLRM model code, which is open-sourced by Facebook (https://github.com/facebookresearch/dlrm). Some minor changes were made to access the model predictions.

Calibration Code (calibration_utils.py, calibration_calibrator):
http://ethen8181.github.io/machine-learning/model_selection/prob_calibration/prob_calibration.html#Calibration-Model
