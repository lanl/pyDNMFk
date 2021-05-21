"""Test script for ML based feature recognition. To run this code, first run the dist_pynmfk_2d_swim.py in examples folder,
for k=12-25 and then run this script.
 """

from pyDNMFk.MLFeatureRecognition import *
targetDir = '../results/swim/'
clf = serialize_deserialize_mlp(model_name='../data/convolute7-model-mAM-p.json').from_json()
pred_model = MLFeaturetools(targetDir, clf)
prediction = pred_model.predictStatistics()
print("Estimated value for K is ",prediction)
