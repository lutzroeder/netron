
import io
import json
import os
import pydoc
import re
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def metadata():

    def parse_docstring(docstring):
        headers = []
        lines = docstring.splitlines()
        indentation = min(filter(lambda s: s > 0, map(lambda s: len(s) - len(s.lstrip()), lines)))
        lines = list((s[indentation:] if len(s) > len(s.lstrip()) else s) for s in lines)
        docstring = '\n'.join(lines)
        tag_re = re.compile('(?<=\n)(Args|Arguments|Variables|Fields|Yields|Call arguments|Raises|Examples|Example|Usage|Input shape|Output shape|Returns|References):\n', re.MULTILINE)
        parts = tag_re.split(docstring)
        headers.append(('', parts.pop(0)))
        while len(parts) > 0:
            headers.append((parts.pop(0), parts.pop(0)))
        return headers

    def parse_arguments(arguments):
        result = []
        item_re = re.compile(r'^   ? ?(\*?\*?\w[\w.]*?\s*):\s', re.MULTILINE)
        content = item_re.split(arguments)
        if content.pop(0) != '':
            raise Exception('')
        while len(content) > 0:
            result.append((content.pop(0), content.pop(0)))
        return result

    def convert_code_blocks(description):
        lines = description.splitlines()
        output = []
        while len(lines) > 0:
            line = lines.pop(0)
            if line.startswith('>>>') and len(lines) > 0 and (lines[0].startswith('>>>') or lines[0].startswith('...')):
                output.append('```')
                output.append(line)
                while len(lines) > 0 and lines[0] != '':
                    output.append(lines.pop(0))
                output.append('```')
            else:
                output.append(line)
        return '\n'.join(output)

    def remove_indentation(value):
        lines = value.splitlines()
        indentation = min(map(lambda s: len(s) - len(s.lstrip()), filter(lambda s: len(s) > 0, lines)))
        lines = list((s[indentation:] if len(s) > 0 else s) for s in lines)
        return '\n'.join(lines).strip()

    def update_argument(schema, name, description):
        if not 'attributes' in schema:
            schema['attributes'] = []
        attribute = next((attribute for attribute in schema['attributes'] if attribute['name'] == name), None)
        if not attribute:
            attribute = {}
            attribute['name'] = name
            schema['attributes'].append(attribute)
        attribute['description'] = remove_indentation(description)

    def update_input(schema, description):
        if not 'inputs' in schema:
            schema['inputs'] = [ { name: 'input' } ]
        parameter = next((parameter for parameter in schema['inputs'] if (parameter['name'] == 'input' or parameter['name'] == 'inputs')), None)
        if parameter:
            parameter['description'] = remove_indentation(description)
        else:
            raise Exception('')

    def update_output(schema, description):
        if not 'outputs' in schema:
            schema['outputs'] = [ { name: 'output' } ]
        parameter = next((parameter for parameter in schema['outputs'] if parameter['name'] == 'output'), None)
        if parameter:
            parameter['description'] = remove_indentation(description)
        else:
            raise Exception('')

    def update_examples(schema, value):
        if 'examples' in schema:
            del schema['examples']
        value = convert_code_blocks(value)
        lines = value.splitlines()
        code = []
        summary = []
        while len(lines) > 0:
            line = lines.pop(0)
            if len(line) > 0:
                if line.startswith('```'):
                    while len(lines) > 0:
                        line = lines.pop(0)
                        if line == '```':
                            break
                        code.append(line)
                else:
                    summary.append(line)
            if len(code) > 0:
                example = {}
                if len(summary):
                    example['summary'] = '\n'.join(summary)
                example['code'] = '\n'.join(code)
                if not 'examples' in schema:
                    schema['examples'] = []
                schema['examples'].append(example)
                code = []
                summary = []

    def update_references(schema, value):
        if 'references' in schema:
            del schema['references']
        references = []
        reference = ''
        lines = value.splitlines()
        for line in lines:
            if line.lstrip().startswith('- '):
                if len(reference) > 0:
                    references.append(reference)
                reference = line.lstrip().lstrip('- ')
            else:
                if line.startswith('  '):
                    line = line[2:]
                reference = ' '.join([ reference, line.strip() ])
        if len(reference) > 0:
            references.append(reference)
        for reference in references:
            if not 'references' in schema:
                schema['references'] = []
            schema['references'].append({ 'description': reference })

    json_path = os.path.join(os.path.dirname(__file__), '../source/keras-metadata.json')
    json_file = open(json_path)
    json_root = json.loads(json_file.read())
    json_file.close()

    for entry in json_root:
        name = entry['name']
        schema = entry['schema']
        if 'package' in schema:
            class_name = schema['package'] + '.' + name
            class_definition = pydoc.locate(class_name)
            if not class_definition:
                raise Exception('\'' + class_name + '\' not found.')
            docstring = class_definition.__doc__
            if not docstring:
                raise Exception('\'' + class_name + '\' missing __doc__.')
            headers = parse_docstring(docstring)
            for header in headers:
                key = header[0]
                value = header[1]
                if key == '':
                    description = convert_code_blocks(value)
                    schema['description'] = remove_indentation(description)
                elif key == 'Args' or key == 'Arguments':
                    arguments = parse_arguments(value)
                    for argument in arguments:
                        update_argument(schema, argument[0], argument[1])
                elif key == 'Call arguments':
                    pass
                elif key == 'Returns':
                    pass
                elif key == 'Input shape':
                    update_input(schema, value)
                elif key == 'Output shape':
                    update_output(schema, value)
                elif key == 'Example' or key == 'Examples' or key == 'Usage':
                    update_examples(schema, value)
                elif key == 'References':
                    update_references(schema, value)
                elif key == 'Variables':
                    pass
                elif key == 'Raises':
                    pass
                else:
                    raise Exception('')

    json_file = open(json_path, 'w')
    json_data = json.dumps(json_root, sort_keys=True, indent=2)
    for line in json_data.splitlines():
        json_file.write(line.rstrip() + '\n')
    json_file.close()

def zoo():
    def download_model(type, file):
        file = os.path.expandvars(file)
        if not os.path.exists(file):
            folder = os.path.dirname(file)
            if not os.path.exists(folder):
                os.makedirs(folder)
            model_type = pydoc.locate(type)
            model = model_type(weights=None)
            model.save(file)
    if not os.environ.get('test'):
        os.environ['test'] = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test'))
    download_model('tensorflow.keras.applications.DenseNet121', '${test}/data/keras/DenseNet121.h5')
    download_model('tensorflow.keras.applications.InceptionResNetV2', '${test}/data/keras/InceptionResNetV2.h5')
    download_model('tensorflow.keras.applications.InceptionV3', '${test}/data/keras/InceptionV3.h5')
    download_model('tensorflow.keras.applications.MobileNetV2', '${test}/data/keras/MobileNetV2.h5')
    download_model('tensorflow.keras.applications.NASNetMobile', '${test}/data/keras/NASNetMobile.h5')
    download_model('tensorflow.keras.applications.ResNet50', '${test}/data/keras/ResNet50.h5')
    download_model('tensorflow.keras.applications.VGG19', '${test}/data/keras/VGG19.h5')

if __name__ == '__main__':
    command_table = { 'metadata': metadata, 'zoo': zoo }
    command = sys.argv[1]
    command_table[command]()
