# A layered sEMG-FMG sensor unit
Code for the machine learning methods and LFCNN in the paper: A layered sEMG-FMG hybrid sensor for hand motion recognition from forearm muscle activities
# Structure of the layerd sEMG-FMG hybrid sensor unit
![overall structure](figure/the-layered-sensor-unit.png)
# LayerFusionNet
![overall structure](figure/layerfusionnet.png)

# Dataset

>We evaluate the performance of the layered sensor unit and the LayerFusionNet on our dataset, some of our dataset you can see from `a-layered-sensor-unit/electrode_position_survey_experiment/data`

# Requirements

* Python 3.8
* Pytorch 1.11.0
* sklearn 0.24.0


# Function of file

* `a-layered-sensor-unit/main_experiment/model/ml/`
  * train machine learning model (XGBoost, SVM, RandomForest, KNN).
* `a-layered-sensor-unit/main_experiment/model/dl/model.py`
  * Generate sEMG-FMG LFN model, sEMG LFN model and FMG LFN model.

# Usage
We've offered three models:  `sEMG-FMG LayerFusionModel` , `sEMG LayerFusionModel` and `FMG LayerFusionModel` for dual modal (sEMG and FMG) and single modal (sEMG or FMG) respectively.
You need to use a tensor with shape: **[Batch_size, channel, length]** for all the three models.
