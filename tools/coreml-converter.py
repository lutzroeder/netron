#!/usr/bin/env python

import os
import sys

file = sys.argv[1];
base, extension = os.path.splitext(file)

if extension == '.h5':
    import coremltools
    coreml_model = coremltools.converters.keras.convert(file)
    coreml_model.save(base + '.mlmodel')
