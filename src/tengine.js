/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var tengine = tengine || {};
var base = base || require('./base');


let buffers = [];
let tensors = [];
let modelLayout = 0;
let origFormat = 0;

var modelVersion = {
    V2 : 2,
    v1 : 1
}

var dataFormat = {
    NCHW : 0,
    NHWC : 1
}

var modelFormat = {
    unknown : 0,
    Tengine : 1,
    Caffe : 2,
    ONNX : 3,
    MxNet : 4,
    TensorFlow : 5,
    Tflite : 6,
    DarkNet : 7,
    DLA : 8

}

var opType = {
    Accuracy : 0,
    BatchNormalization : 1,
    BilinearResize : 2,
    Concat : 3,
    Const : 4,
    Convolution : 5,
    DeConvolution : 6,
    DetectionOutput : 7,
    DropOut : 8,
    Eltwise : 9,
    Flatten : 10,
    FullyConnected : 11,
    INPUT : 12,
    LRN : 13,
    Normalize : 14,
    Permute : 15,
    Pooling : 16,
    Prelu : 17,
    PriorBox : 18,
    Region : 19,
    ReLU : 20,
    ReLU6 : 21,
    Reorg : 22,
    Reshape : 23,
    RoiPooling : 24,
    RPN : 25,
    Scale : 26,
    Slice : 27,
    SoftMax : 28,
    Split : 29,
    DetectionPostProcess : 30,
    Gemm : 31,
    Generic : 32,
    Logistic : 33,
    LSTM : 34,
    RNN : 35,
    TanH : 36,
    Sigmoid : 37,
    Squeeze : 38,
    FusedbnScaleRelu : 39,
    Pad : 40,
    StridedSlice : 41,
    ArgMax : 42,
    ArgMin : 43,
    TopKV2 : 44,
    Reduction : 45,
    Max : 46,
    Min : 47,
    GRU : 48,
    Addn : 49,
    SwapAxis : 50,
    Upsample : 51,
    SpaceToBatchND : 52,
    BatchToSpaceND : 53,
    Resize : 54,
    ShuffleChannel : 55,
    Crop : 56,
    ROIAlign : 57,
    Psroipooling : 58,
    Unary : 59,
    Expanddims : 60,
    Bias : 61,
    Noop : 62,
    Threshold : 63,
    Hardsigmoid : 64,
    Embed : 65,
    InstanceNorm : 66,
    MVN : 67,
    Absval : 68,
    Cast : 69,
    HardSwish : 70,
    Interp : 71,
    SELU : 72, 
    ELU : 73,
    BroadMul : 74,
    Logical : 75,
    Gather : 76,
    Transpose : 77,
    Num : 78

}

tengine.ModelFactory = class {

    match(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.tmfile')) {
            const buffer = context.buffer;
            if (buffer.length > 4) {
                const mainVer = buffer[0] | buffer[1] << 8 ;
                if(mainVer === 0x0002 | mainVer === 0x0001) 
                return true;
                            }
        }   
        return false;
    }

    open(context, host) {
        return tengine.Metadata.open(host).then((metadata) => {
            const identifier = context.identifier.toLowerCase();
            const tmfile = (tmfile) => {
                try {
                    return new tengine.Model(metadata, tmfile);
                }
                catch (error) {
                    let message = error && error.message ? error.message : error.toString();
                    message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
                    throw new tengine.Error(message + " in '" + identifier + "'.");
                }
            };
            return tmfile(context.buffer);  
        });
    }
}

tengine.Model = class {

    constructor(metadata,tmfile) {
        this._graphs = [];
        this._graphs.push(new tengine.Graph(metadata, tmfile)); 

        this._header = tmfile;
        this._mainVer = ((this._header[0] | this._header[1] << 8 ) == 0x0002) ? 2:1;
        
    }

    get format() {
        return "Tengine";
    }

    get dataFormat(){
        
        if(modelLayout == dataFormat.NCHW)
            return "NCHW";
        else if(modelLayout == dataFormat.NHWC)
            return "NHWC";
    }

    get origFormat(){   
        let rootTable = (this._header[8] | this._header[9] << 8 | this._header[10] << 16 | this._header[11] << 24);
        origFormat= (this._header[rootTable] | this._header[rootTable+1] << 8 
            | this._header[rootTable+2] << 16 | this._header[rootTable+3] << 24);
        let format;
        switch(origFormat){
            case modelFormat.Tengine:
                format = 'Tengine';
                break;
            case modelFormat.Caffe:
                format = 'Caffe';
                break;
            case modelFormat.ONNX:
                format = 'ONNX';
                break;
            case modelFormat.MxNet:
                format = 'MxNet';
                break;
            case modelFormat.TensorFlow:
                format = 'TensorFlow';
                break;
            case modelFormat.Tflite:
                format = 'Tflite';
                break;
            case modelFormat.DarkNet:
                format = 'DarkNet';
                break;
            case modelFormat.DLA:
                format = 'DLA';
                break;
            case modelFormat.unknown:
                format = 'unknown';
                break;
        }
        return format;
        
    }
    
    get version() {
      if(this._mainVer == modelVersion.V2)
            return 'TMFILE V2';
        else if(this._mainVer == modelVersion.v1)
            return 'TMFILE V1';
        else   
            return false;
    }   


    get graphs() {
        return this._graphs; 
    }
}

tengine.Graph = class {

    constructor(metadata,tmfile) {
        this._inputs = [];
        this._outputs = [];
        this._nodes = [];

        let layers = this._tmfile_bin(tmfile);
        const data2 = new tengine.BlobReader(tmfile);

         for (const layer of layers) {
            if (layer.type == opType.INPUT ) {
                let dimensions = [];
                dimensions = tensors[layer.output_tensorID].dims;
                for(let i =0; i< dimensions.length; i++){
                    if(dimensions[i] == -1 ){           // -1 == ?
                    dimensions.shift();
                    }
                }
                if(dimensions[0] == 1)
                    dimensions.shift();
                const shape = new tengine.TensorShape(dimensions);
                const type = new tengine.TensorType('float32', shape);
                this._inputs.push(new tengine.Parameter(layer.name, true, layer.outputs.map((output) => new tengine.Argument(output, type, null))));
            }
            else {
                this._nodes.push(new tengine.Node2(metadata,layer,data2));
            }
        }        
    }

    _tmfile_bin(tmfile){ 
        let layers = [];
        const tm2_model = new tengine.tmfileReader(tmfile);
        let rootTable = tm2_model.read4Bytes(8);           // offset to Root Table (model)

        // model is Root Table, contains format and subgraph info
        let model = {};

        model.subgraphVecOffset = tm2_model.read4Bytes(rootTable+8);     // offset to vector of subgraphs 

        // subgraphVes is the vector to subgraphs, contains sizes and offsets
        let subgraphVec = {};
        subgraphVec.size = tm2_model.read4Bytes(model.subgraphVecOffset);      // Only one subgraph supported right now 

        if(subgraphVec.size != 1)
            return false;
        subgraphVec.addr = tm2_model.read4Bytes(model.subgraphVecOffset+4);    //
     
        // subgraph is the TM_Subgraph object, contains nodes vec, tensors vec, buffers vec
        let subgraph = {};
        subgraph.id = tm2_model.read4Bytes(subgraphVec.addr);
        subgraph.graphLayout = tm2_model.read4Bytes(subgraphVec.addr+4);    // layout, 1 for NHWC, 2 for NCHW
        modelLayout = subgraph.graphLayout;
        subgraph.nodesVecOffset = tm2_model.read4Bytes(subgraphVec.addr+20);      // offset to vector of nodes
        subgraph.tensorsVecOffset = tm2_model.read4Bytes(subgraphVec.addr+24);    // offset to vector of tensors
        subgraph.buffersVecOffset = tm2_model.read4Bytes(subgraphVec.addr+28);    // offset to vector of buffers
      
        const tm_nodesCount = tm2_model.read4Bytes(subgraph.nodesVecOffset);      // The number of nodes from the vector of nodes 
        const tm_tensorCount = tm2_model.read4Bytes(subgraph.tensorsVecOffset);   // The number of tensors from the vector of tensors
        const tm_buffersCount = tm2_model.read4Bytes(subgraph.buffersVecOffset);  // The number of buffers from the vector of buffers
   
        // Read all nodes address into nodesAddr[], all nodes into nodes[]
        let nodesAddr = [];
        let nodes = [];
        for(let i=0; i<tm_nodesCount; i++){
            let node = {};
            nodesAddr.push(tm2_model.read4Bytes(subgraph.nodesVecOffset + 4*(i+1)));
            node.id = tm2_model.read4Bytes(nodesAddr[i]);
            node.inputTensorsOffset = tm2_model.read4Bytes(nodesAddr[i]+4);            // The offset to the vector of input tensors 
            node.outputTensorsOffset = tm2_model.read4Bytes(nodesAddr[i]+8);           // The offset to the vector of output tensors
            node.operatorOffset = tm2_model.read4Bytes(nodesAddr[i]+12);               // The offset to TM_Operator
            node.nodeNameOffset = tm2_model.read4Bytes(nodesAddr[i]+16);               // The offset to TM_String
            node.attrsVecOffset = tm2_model.read4Bytes(nodesAddr[i]+20);               // The offset to the vector of attributes (always 0)
            node.dynamicShape = tm2_model.readBool(nodesAddr[i]+24) ? true : false;    // If dynamic shape? Bool
            
            // get the name of this node into node.name
            let nodeNameSize = tm2_model.read4Bytes(node.nodeNameOffset);               // From the TM_String, first the size of this string
            let nodeNameAddr = tm2_model.read4Bytes(node.nodeNameOffset+4);
            node.name = tm2_model.readString(nodeNameAddr, nodeNameSize);
            
            // get all the input tensor names into node.input[]
            node.input = [];

            let nodeInputSize = node.inputTensorsOffset ? tm2_model.read4Bytes(node.inputTensorsOffset) : 0;
            for (let j = 0; j<nodeInputSize; j++){
                node.input.push(tm2_model.read4Bytes(node.inputTensorsOffset+(j+1)*4));   // just number -- tensor[number]
            }

            // get all the output tensor names into node.output[]
            node.output = [];
            let nodeOutputSize = tm2_model.read4Bytes(node.outputTensorsOffset);
            for (let j = 0; j<nodeOutputSize; j++){
                let address_temp = (node.outputTensorsOffset + 4*(j+1));
                //node.output.push(tm2_model.read4Bytes(node.OutputTensorsOffset + (j+1)*4 ) );   // just number -- tensor[number]
                node.output.push(tm2_model.read4Bytes(address_temp) );   // just number -- tensor[number]
            }

            node.opType = tm2_model.read4Bytes(node.operatorOffset+4);
            node.paramAddr = tm2_model.read4Bytes(node.operatorOffset+8);
            // if(node.opType == 5) {
            //     node.opParam = tm2_model.readParams(node.paramAddr,14);
            // } 

             
            switch(node.opType){
                case opType.BatchNormalization:
                    node.attrCount = 3;
                    break;
                case opType.BilinearResize:
                    node.attrCount = 3;
                    break;
                case opType.Concat:
                    node.attrCount = 1;
                    break;
                case opType.Convolution:
                    node.attrCount = 14;
                    break;
                case opType.DeConvolution:
                    node.attrCount = 13;
                    break;
                case opType.DetectionOutput:
                    node.attrCount = 5;
                    break;
                case opType.Eltwise:
                    node.attrCount = 2;
                    break;
                case opType.Flatten:
                    node.attrCount = 2;
                    break;
                case opType.FullyConnected:
                    node.attrCount = 1;
                    break;
                case opType.LRN:
                    node.attrCount = 5;
                    break;
                case opType.Normalize:
                    node.attrCount = 2;
                    break;
                case opType.Permute:
                    node.attrCount = 5;
                    break;
                case opType.Pooling:
                    node.attrCount = 11;
                    break;
                case opType.PriorBox:
                    node.attrCount = 14;
                    break;
                case opType.Region:
                    node.attrCount = 7;
                    break;
                case opType.ReLU:
                    node.attrCount = 1;
                    break;
                case opType.Reorg:
                    node.attrCount = 1;
                    break;
                case opType.Reshape:
                    node.attrCount = 3;
                    break;
                case opType.RoiPooling:
                    node.attrCount = 3;
                    break;
                case opType.RPN:
                    node.attrCount = 9;
                    break;
                case opType.Scale:
                    node.attrCount = 3;
                    break;
                case opType.Slice:
                    node.attrCount = 8;
                    break;
                case opType.SoftMax:
                    node.attrCount = 1;
                    break;
                case opType.DetectionPostProcess:
                    node.attrCount = 6;
                    break;
                case opType.Gemm:
                    node.attrCount = 4;
                    break;
                case opType.Generic:
                    node.attrCount = 3;
                    break;
                case opType.LSTM:
                    node.attrCount = 18;
                    break;
                case opType.RNN:
                    node.attrCount = 9;
                    break;
                case opType.Squeeze:
                    node.attrCount = 4;
                    break;
                case opType.Pad:
                    node.attrCount = 10;
                    break;
                case opType.StridedSlice:
                    node.attrCount = 12;
                    break;
                case opType.ArgMax:
                    node.attrCount = 1;
                    break;
                case opType.ArgMin:
                    node.attrCount = 1;
                    break;
                case opType.TopKV2:
                    node.attrCount = 2;
                    break;
                case opType.Reduction:
                    node.attrCount = 6;
                    break;
                case opType.GRU:
                    node.attrCount = 10;
                    break;
                case opType.Addn:
                    node.attrCount = 1;
                    break;
                case opType.SwapAxis:
                    node.attrCount = 2;
                    break;
                case opType.Upsample:
                    node.attrCount = 1;
                    break;
                case opType.SpaceToBatchND:
                    node.attrCount = 6;
                    break;
                case opType.BatchToSpaceND:
                    node.attrCount = 6;
                    break;
                case opType.Resize:
                    node.attrCount = 3;
                    break;
                case opType.ShuffleChannel:
                    node.attrCount = 1;
                    break;
                case opType.Crop:
                    node.attrCount = 9;
                    break;
                case opType.ROIAlign:
                    node.attrCount = 3;
                    break;
                case opType.Psroipooling:
                    node.attrCount = 4;
                    break;
                case opType.Unary:
                    node.attrCount = 1;
                    break;
                case opType.Expanddims:
                    node.attrCount = 1;
                    break;
                case opType.Bias:
                    node.attrCount = 1;
                    break;
                case opType.Threshold:
                    node.attrCount = 1;
                    break;
                case opType.Hardsigmoid:
                    node.attrCount = 2;
                    break;
                case opType.Embed:
                    node.attrCount = 4;
                    break;
                case opType.InstanceNorm:
                    node.attrCount = 1;
                    break;
                case opType.MVN:
                    node.attrCount = 3;
                    break;
                case opType.Cast:
                    node.attrCount = 2;
                    break;
                case opType.HardSwish:
                    node.attrCount = 2;
                    break;
                case opType.Interp:
                    node.attrCount = 5;
                    break;
                case opType.SELU:
                    node.attrCount = 2;
                    break;
                case opType.ELU:
                    node.attrCount = 1;
                    break;
                case opType.Logical:
                    node.attrCount = 1;
                    break;
                case opType.Gather:
                    node.attrCount = 2;
                    break;
                case opType.Transpose:
                    node.attrCount = 1;
                    break;
            }
            if(node.opType != opType.Accuracy || opType.Const || 
                opType.DropOut || opType.Input || opType.Prelu || 
                opType.ReLU6 || opType.Split || opType.Logistic || 
                opType.TanH || opType.Sigmoid || opType.FusedbnScaleRelu || 
                opType.Max || opType.Min || opType.Noop || 
                opType.Absval || opType.BroadMul || opType.Num){
                    node.opParam = tm2_model.readParams(node.paramAddr,node.attrCount);
                }
            nodes.push(node);
        }

        // Read all tensors address into tensorsAddr[], all tensors into tensors[]
        let tensorsAddr = [];
        for (let i = 0; i < tm_tensorCount ; i++){
            let tensor = {};
            tensorsAddr.push(tm2_model.read4Bytes(subgraph.tensorsVecOffset + 4*(i+1)));
            tensor.id = tm2_model.read4Bytes(tensorsAddr[i]);
            tensor.bufferId = tm2_model.read4Bytes(tensorsAddr[i]+4);                  // The buffer ID used by this tensor
            tensor.dimsVecOffset = tm2_model.read4Bytes(tensorsAddr[i]+8);             // The offset to the vector of dims
            tensor.tensorNameOffset = tm2_model.read4Bytes(tensorsAddr[i]+12);         // The offset to the string of tensor name
            
            // get the name of this tensor
            let tensorNameSize = tm2_model.read4Bytes(tensor.tensorNameOffset);
            let tensorNameAddr = tm2_model.read4Bytes(tensor.tensorNameOffset+4);
            tensor.name = tm2_model.readString(tensorNameAddr, tensorNameSize);        // The name of this tensor into tensor.name

            tensor.quantParamsOffset = tm2_model.read4Bytes(tensorsAddr[i]+16);        // The offset to the quant params of the tensor, 0 for QUANT_FP16; 1 for QUANT_INT8; 2 for QUANT_UINT8
            tensor.layout = tm2_model.read4Bytes(tensorsAddr[i]+20);                   // The layout of this tensor
            tensor.type = tm2_model.read4Bytes(tensorsAddr[i]+24);                     // The type of this tensor
            tensor.dataType = tm2_model.read4Bytes(tensorsAddr[i]+28);                 // The data type of this tensor, 
            
            // get the dims into tensor.dims
            tensor.dims = [];
            if(tensor.dimsVecOffset != 0x0000){
            tensor.dimsNum = tm2_model.read4Bytes(tensor.dimsVecOffset);                // The number of dimensions
            for (let j = 0; j<tensor.dimsNum; j++){
                tensor.dims.push(tm2_model.read4Bytes(tensor.dimsVecOffset+(j+1)*4));       // The elements of every dim
                }
            }

            tensors.push(tensor);
        }

        // Read all buffers address into buffersAddr[], all tensors into buffers[]
        let buffersAddr = [];
        // let buffers = [];
        for (let i = 0; i < tm_buffersCount ; i++){
            let buffer = {};
            buffersAddr.push(tm2_model.read4Bytes(subgraph.buffersVecOffset + 4*(i+1)));
            buffer.size = tm2_model.read4Bytes(buffersAddr[i]);                         // The size of the buffer[i]
            buffer.offset = tm2_model.read4Bytes(buffersAddr[i]+4);                     // The offset to the real buffer from the vector of buffer[i]
            buffer.data = tm2_model.readBuffers(buffer.offset, buffer.size);          // The buffer data of the buffer[i]
            buffers.push(buffer);
        }
        
        for(let i = 0; i< tm_nodesCount; i++)
        {
            let layer = {};
            layer.name =  nodes[i].name;                                // tensors name []
            
            // The input of this layer
            layer.inputs = [];
            let inputCount = nodes[i].input.length;
            for(let j = 0; j< inputCount; j++){
                let name_temp = tensors[nodes[i].input[j]].name;
                 if(j ==1)
                layer.input_tensorID = tensors[nodes[i].input[1]].id;
                if(j ==0)
                layer.input2_tensorID = tensors[nodes[i].input[0]].id;
                layer.inputs.push(name_temp);
            }

            // The outputs of this layer
            layer.outputs = [];
            let outputCount = nodes[i].output.length;
            for(let j = 0; j< outputCount; j++){
                let name_temp = tensors[nodes[i].output[j]].name;
                layer.output_tensorID = tensors[nodes[i].output[0]].id;
                layer.outputs.push(name_temp);
                
            }

            layer.type = nodes[i].opType;

            let attr = {};
            layer.attributes = [];

            if(layer.type != opType.Accuracy || opType.Const || 
                    opType.DropOut || opType.Input || opType.Prelu || 
                    opType.ReLU6 || opType.Split || opType.Logistic || 
                    opType.TanH || opType.Sigmoid || opType.FusedbnScaleRelu || 
                    opType.Max || opType.Min || opType.Noop || 
                    opType.Absval || opType.BroadMul || opType.Num) {
                for (let t = 0; t<nodes[i].attrCount; t++){
                    attr = {key: t, value: nodes[i].opParam[t]};
                    layer.attributes.push(attr);
                }
            }

            if(layer.type == opType.Convolution){
                if(modelLayout == 1){
                    attr = {key: 6, value: tensors[layer.input_tensorID].dims[3]};      // NHWC
                    }
                else{
                    attr = {key: 6, value: tensors[layer.input_tensorID].dims[1]};      // NCHW
                }
                layer.attributes[6] = attr;
            }

            if(layer.type == opType.Slice){ // [4]--iscaffe [5]--ismxnet
                attr = (nodes[i].opParam[4] == 1) ?  // [4] -- iscaffe
                {key: 4, value: 1 } : {key: 4, value: 0};
                layer.attributes[4] = attr;
                attr = (nodes[i].opParam[5] == 1) ?  // [5] -- ismxnet
                {key: 5, value: 1} : {key: 5, value: 0};
                layer.attributes[5] = attr;
                attr = (origFormat == modelFormat.TensorFlow) ?  // [5] -- ismxnet
                {key: 6, value: nodes[i].opParam[6]} : {key: 6, value: 0};
                layer.attributes[6] = attr;
            }

            if(layer.type == opType.Reshape){
                attr = (nodes[i].opParam[0] == 1) ?  // [0] -- isMxNet
                {key: 0, value: 1 } : {key: 0, value: 0};
            }
            
            if(layer.type != opType.Const)
                layers.push(layer);
        }
        return layers;
    }


    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
}

tengine.Parameter = class {

    constructor(name, visible, args) {              //(inputName, true, argument(input, null, null))
        this._name = name;
        this._visible = visible;                    // True
        this._arguments = args;                     // tengine.Argument(input, null, null)
    }

    get name() {
        return this._name;
    }

    get visible() {
        return this._visible;
    }

    get arguments() {
        return this._arguments;
    }
};

tengine.Argument = class { 

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};


tengine.Node2 = class {

    constructor(metadata, layer,data2) {
        this._metadata = metadata;
        this._inputs = [];
        this._outputs = [];
        this._attributes = [];
        this._operator = layer.type; 
        this._name = layer.name;
        this._tensorIn = layer.input_tensorID;
        this._tensorIn2 = layer.input2_tensorID;
        this._tensorOut = layer.output_tensorID;
        this._is
        // console.log("The tensor's id is: %s",this._tensorIn);

        const operator = metadata.getOperatorName(this._operator);
        
        if (operator) {
            this._operator = operator;
        }

        const schema = metadata.type(this._operator);

        let attributeMetadata = {};
        if (schema && schema.attributes) { 
            for (let i = 0; i < schema.attributes.length; i++) { 
                const id = schema.attributes[i].id || i.toString(); 
                attributeMetadata[id] = schema.attributes[i]; 
            }
        }
        for (const attribute of layer.attributes) {
            const attributeSchema = attributeMetadata[attribute.key];
            this._attributes.push(new tengine.Attribute(attributeSchema, attribute.key, attribute.value)); 
        }

        let inputs = layer.inputs; 
        let inputIndex = 0;
        if (schema && schema.inputs) { 
            for (const inputDef of schema.inputs) {
                if (inputIndex < inputs.length || inputDef.option != 'optional') {
                    let inputCount = (inputDef.option == 'variadic') ? (inputs.length - inputIndex) : 1;
                    let inputArguments = inputs.slice(inputIndex, inputIndex + inputCount).filter((id) => id != '' || inputDef.option != 'optional').map((id) => {
                        return new tengine.Argument(id, null, null);
                    });
                    this._inputs.push(new tengine.Parameter(inputDef.name, true, inputArguments));
                    inputIndex += inputCount;
                }
            }
        }
        else {
            this._inputs = this._inputs.concat(inputs.slice(inputIndex).map((input, index) => { 
                let inputName = ((inputIndex + index) == 0) ? 'input' : (inputIndex + index).toString(); 
                return new tengine.Parameter(inputName, true, [
                    new tengine.Argument(input, null, null) 
                ]);
            }));
        }

        let outputs = layer.outputs;
        let outputIndex = 0;
        if (schema && schema.outputs) {
            for (const outputDef of schema.outputs) { 
                if (outputIndex < outputs.length || outputDef.option != 'optional') {
                    let outputCount = (outputDef.option == 'variadic') ? (outputs.length - outputIndex) : 1;
                    let outputArguments = outputs.slice(outputIndex, outputIndex + outputCount).map((id) => {
                        return new tengine.Argument(id, null, null)
                    });
                    this._outputs.push(new tengine.Parameter(outputDef.name, true, outputArguments));
                    outputIndex += outputCount;
                }
            }
        }
        else {
            this._outputs = this._outputs.concat(outputs.slice(outputIndex).map((output, index) => {
                let outputName = ((outputIndex + index) == 0) ? 'output' : (outputIndex + index).toString();
                return new tengine.Parameter(outputName, true, [
                    new tengine.Argument(output, null, null)
                ]);
            }));
        }

        switch (this._operator) {
            
            case 'Convolution':{
                this._weight('filters',[tensors[this._tensorIn].dims[0],tensors[this._tensorIn].dims[1],
                tensors[this._tensorIn].dims[2],tensors[this._tensorIn].dims[3]],'float32',data2);
                break;
            }
            
            case 'PReLU': {
                break;
            }
        }
    }

    get operator() {
        return this._operator;
    }

    get name() {
        return this._name;
    }

    get category() {
        const schema = this._metadata.type(this._operator);
        return (schema && schema.category) ? schema.category : '';
    }

    get documentation() {
        return '';
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    _weight( name, dimensions, dataType, data2) {        
        const blob = data2.read(dimensions, dataType);
        // console.log("%o",blob);
        this._inputs.push(new tengine.Parameter(name, true, [
            new tengine.Argument('', null, new tengine.Tensor(new tengine.TensorType(dataType, new tengine.TensorShape(dimensions)), blob ))
        ]));
    }
}

tengine.Attribute = class {

    constructor(schema, key, value) {
        this._type = '';
        this._name = key;
        this._value = value;
        if (schema) {
            this._name = schema.name;
            if (schema.type) {
                this._type = schema.type;
            }
            switch (this._type) {
                case 'int32':
                    this._value = parseInt(this._value, 10);  
                    break;
                case 'float32':
                    this._value = parseFloat(this._value);
                    break;
                case 'float32[]':
                    this._value = this._value.map((v) => parseFloat(v));
                    break;
            }
            if (Object.prototype.hasOwnProperty.call(schema, 'visible') && !schema.visible) {
                this._visible = false;
            }
            else if (Object.prototype.hasOwnProperty.call(schema, 'default')) {
                if (this._value == schema.default || (this._value && this._value.toString() == schema.default.toString())) {
                    this._visible = false;
                }
            }
        }
    }

    get type() {
        return this._type;
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
}

tengine.Tensor = class {

    constructor(type, data) {
        this._type = type;
        this._data = data;
    }

    get kind() {
        return 'Weight';
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state || null;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        context.index = 0;
        context.count = 0;
        context.state = null;

        if (this._type.dataType == '?') {
            context.state = 'Tensor has unknown data type.';
            return context;
        }
        if (!this._type.shape) {
            context.state = 'Tensor has no dimensions.';
            return context;
        }

        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        switch (this._type.dataType) {
            case 'float16':
            case 'float32':
                context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);
                break;
            default:
                context.state = 'Tensor data type is not implemented.';
                break;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (context.shape.length == 0) {
            shape = [ 1 ];
        }
        let results = [];
        let size = shape[dimension];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (this._type.dataType) {
                    case 'float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }

}

tengine.TensorType = class {

    constructor(dataType, shape) {
        this._dataType = dataType || '?';
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this._dataType + this._shape.toString();
    }
}

tengine.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        return this._dimensions ? ('[' + this._dimensions.map((dimension) => dimension ? dimension.toString() : '?').join(',') + ']') : '';
    }
};

tengine.Metadata = class {

    static open(host) {
        if (tengine.Metadata._metadata) {
            return Promise.resolve(tengine.Metadata._metadata);
        }
        return host.request(null, 'tengine-metadata.json', 'utf-8').then((data) => {
            tengine.Metadata._metadata = new tengine.Metadata(data);
            return tengine.Metadata._metadata;
        }).catch(() => {
            tengine.Metadata._metadata = new tengine.Metadata(null);
            return tengine.Metadata._metadatas;
        });
    }

    constructor(data) {
        this._operatorMap = {}; 
        this._map = {};
        this._attributeCache = {};
        if (data) {
            let items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                        if (Object.prototype.hasOwnProperty.call(item.schema, 'operator')) {
                            this._operatorMap[item.schema.operator.toString()] = item.name;
                        }
                    }
                }
            }
        }
    }

    getOperatorName(code) {
        return this._operatorMap[code] || null;
    }

    type(operator) {
        return this._map[operator] || null;
    }

    attribute(operator, name) {
        let map = this._attributeCache[operator];
        if (!map) {
            map = {};
            const schema = this.type(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                for (const attribute of schema.attributes) {
                    map[attribute.name] = attribute;
                }
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

tengine.tmfileReader = class {
    constructor(tmfile){
        this._header = tmfile;
    }

    readBool(offset){
        let position = offset;
        let f0 = this._header[position++];
        this._bool = f0 & 0x8;
    }
    
    read4Bytes(offset){
        let position = offset;
        let f0 = this._header[position++];
        let f1 = this._header[position++];
        let f2 = this._header[position++];
        let f3 = this._header[position++];
        this._read4Bytes = f0 | f1 << 8 | f2 << 16 | f3 << 24;
        return this._read4Bytes;
    }

    read2Bytes(offset){
        let position = offset;
        let f0 = this._header[position++];
        let f1 = this._header[position++];
        this._read2Bytes = f0 | f1 << 8 ;
        return this._read2Bytes;
    }

    read1Byte(offset){
        return this._header[offset];
    }

    readBuffers(offset,size){                               
        let position = offset;
        let uint8Array = new Uint8Array(size);
        for(let i=0; i < (size); i++){
            uint8Array[i++] = this._header[position++];
        }
        return uint8Array;
    }

    readParams(offset,size){                               
        let param = [];
        let position = offset;
        for(let i=0; i < size; i++){
        let f0 = this._header[position++];
        let f1 = this._header[position++];
        let f2 = this._header[position++];
        let f3 = this._header[position++];
        param.push(f0 | f1 << 8 | f2 << 16 | f3 << 24);
        }
        return param;
    }

    readString(offset,size){
        let string = [];
        let position = offset;
        for(let i=0; i < size; i++){
        string.push(String.fromCharCode(this._header[position++]));
        }
        return string.join("").toString();
    }

}


tengine.BlobReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._position = 0;
    }

    read(shape, dataType) {
        if (this._buffer) {
            let data = null;
            let size = 1;
            if (shape) {
                for (const dimension of shape) {
                    size *= dimension;
                }
            }
            else {
                this._buffer = null;
            }
            if (this._buffer) {
                if (dataType) {
                    let position = this._position
                    switch (dataType) {
                        case 'float32': 
                            size *= 4;
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'float16': 
                            size *= 2;
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'int8':
                            this._position += size;
                            data = this._buffer.subarray(position, this._position);
                            break;
                        case 'qint8':
                            this._position += size + 1024;
                            data = null;
                            break;
                        default:
                            throw new tengine.Error("Unknown weight type '" + dataType + "'.");
                    }
                }
            }
            return { dataType: dataType, data: data };
        }
        return null;
    }
}

tengine.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading tengine model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = tengine.ModelFactory;
}
