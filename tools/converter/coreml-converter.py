#!/usr/bin/env python

import os
import sys

file = sys.argv[1];
base, extension = os.path.splitext(file)

if extension == '.h5':
    import coremltools
    coreml_model = coremltools.converters.keras.convert(file)
    coreml_model.save(base + '.mlmodel')
elif extension == '.pkl':
    import coremltools
    import sklearn
    sklearn_model = sklearn.externals.joblib.load(file)
    coreml_model = coremltools.converters.sklearn.convert(sklearn_model)
    coreml_model.save(base + '.mlmodel')
