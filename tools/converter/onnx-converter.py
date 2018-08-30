#!/usr/bin/env python

import os
import sys

file = sys.argv[1];
base, extension = os.path.splitext(file)

if extension == '.mlmodel':
	import coremltools
	import onnxmltools
	coreml_model = coremltools.utils.load_spec(file)
	onnx_model = onnxmltools.convert.convert_coreml(coreml_model)
	onnxmltools.utils.save_model(onnx_model, base + '.onnx')
elif extension == '.h5':
	import keras
	import onnxmltools
	keras_model = keras.models.load_model(file)
	onnx_model = onnxmltools.convert.convert_keras(keras_model)
	onnxmltools.utils.save_model(onnx_model, base + '.onnx')
elif extension == '.pkl':
	from sklearn.externals import joblib
	import onnxmltools
	sklearn_model = joblib.load(file)
	onnx_model = onnxmltools.convert.convert_sklearn(sklearn_model)
	onnxmltools.utils.save_model(onnx_model, base + '.onnx')
