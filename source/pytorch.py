
def serialize(model):
    print('Experimental')
    import torch
    json_model = {}
    json_model['signature'] = 'netron:pytorch'
    json_model['format']  = 'TorchScript'
    json_model['graphs'] = []
    json_graph = {}
    json_graph['nodes'] = []
    json_model['graphs'].append(json_graph)
    for node in model.nodes():
        json_node = {
            'inputs': [],
            'outputs': [],
            'attributes': []
        }
        json_node['type'] = {
            'name': node.kind()
        }
        json_graph['nodes'].append(json_node)
        for name in node.attributeNames():
            value = node[name]
            json_attribute = {
                'name': name,
                'value': value
            }
            if torch.is_tensor(value):
                json_node['inputs'].append({
                    'name': name,
                    'arguments': []
                })
            else:
                json_node['attributes'].append(json_attribute)


        for input in node.inputs():
            name = str(input.unique())
            argument = {
                'name': name
            }
            parameter = {
                'name': 'x',
                'arguments': [ argument ]
            }
            json_node['inputs'].append(parameter)

        for output in node.outputs():
            name = str(output.unique())
            argument = {
                'name': name
            }
            parameter = {
                'name': 'x',
                'arguments': [ argument ]
            }
            json_node['outputs'].append(parameter)

    import json
    text = json.dumps(json_model, ensure_ascii=False, indent=2)
    return text.encode('utf-8')
