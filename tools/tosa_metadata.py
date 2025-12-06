#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

"""Parsing of tosa.xml to tosa_metadata.json"""

import argparse
import json
import re
import sys
from xml.etree import ElementTree as ET

# Add argument parsing
parser = argparse.ArgumentParser(description="TOSA XML to metadata JSON")
parser.add_argument("--xml", help="XML file to parse", required=True)
parser.add_argument("outfile", nargs="?", type=argparse.FileType("w"),
                    default=sys.stdout)
args = parser.parse_args()

# read in the XML file
root = ET.parse(args.xml)

output = []
# append all operators
operators = root.findall(".//operator")
for o in operators:
    name = o.find("name").text
    output.append({
        "name": name,
        "category": "",
        "attributes": [],
        "inputs": [],
        "outputs": []})
    # add category mapping
    if "CONST" in name:
        output[-1]["category"] = "constant"
    elif "CUSTOM" in name:
        output[-1]["category"] = "custom"
    elif "POOL" in name:
        output[-1]["category"] = "pool"
    elif "RESCALE" in name:
        output[-1]["category"] = "quantization"
    elif any(c in name for c in ["SHAPE", "DIM"]):
        output[-1]["category"] = "shape"
    elif any(c in name for c in ["CONV", "FULLY_CONNECTED"]):
        output[-1]["category"] = "layer"
    elif name in ["TRANSPOSE", "GATHER", "SCATTER"]:
        output[-1]["category"] = "transform"
    elif name in ["CLAMP", "SIGMOID", "TANH", "ERF"]:
        output[-1]["category"] = "activation"
    elif name in ["CONCAT", "PAD", "REVERSE", "SLICE", "TILE", "IDENTITY"]:
        output[-1]["category"] = "tensor"
    else:
        del output[-1]["category"]
    arguments = o.findall(".//argument")
    for arg in arguments:
        category = arg.attrib["category"]
        description = re.sub(r"[\s+]+", " ", arg.find("description").text).strip()
        element_type = arg.attrib["tensor-element-type"]
        obj = {
                "name": arg.attrib["name"],
                "type": arg.attrib["type"] if element_type == "-" else element_type,
                "description": description
        }
        # sometimes the same argument can be
        # an input or an attribute, depending on profile
        # so make sure we match on both
        if "attribute" in category:
            output[-1]["attributes"].append(obj)
        if "input" in category:
            output[-1]["inputs"].append(obj)
        elif "output" in category:
            output[-1]["outputs"].append(obj)

# write out the JSON file
json.dump(output, args.outfile, indent=2)
