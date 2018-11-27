Term
    = 
    next0: CommentsTag* _
    meta: XMLVersionTag? _ 
    next1: CommentsTag* _
    first:NetTag _
    next2: CommentsTag* _
    layers: LayersTag _
    next3: CommentsTag* _
    close: LayersCloseTag _
    next4: CommentsTag* _
    edges: EdgesTag _
    next5: CommentsTag* _
    closeEdges: EdgesCloseTag _
    next6: CommentsTag* _
    rest: .*  {
        var key = Object.keys(layers)[0];
        var layers_cleaned = layers.map(function (x,y) {
            return x;
        }).filter(function (x) {
            return x;
        });
        first["layers"] = layers_cleaned;
        first["edges"] = edges[1];
        return first;
    }

XMLVersionTag
    = first:("<\?xml")
    rest: tag_attr_pair* "\?>"

NetTag
    = first:("<net")
    rest: tag_attr_pair* ">" _ {
        var key = first.slice(1);
        var res = rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
        var ret = {};
        ret[key] = res;
        return ret;
    }

// ---------------------------------- LayersTag >>>>>>>>>>>>>>>>>>>>>>>>>

CommentsTag
    = "<!--" inner:(!"-->" i:. {return i})* "-->" {return inner.join('')}

LayersTag
    = first:("<layers>" _ )
    next0: CommentsTag* _
    rest: (_ layer:LayerTag {return layer;})* {
        return rest;
    }

LayersCloseTag
    = ("</layers>")

// ----------------------------------LayersTag <<<<<<<<<<<<<<<<<<<<<<<<<<<

// ---------------------------------- LayerTag >>>>>>>>>>>>>>>>>>>>>>>>>

LayerTag
    = 
    next0: CommentsTag* _
    start: LayerStartTag
    next1: CommentsTag* _
    data: DataTag?
    next2: CommentsTag* _
    input: InputsTag?
    next3: CommentsTag* _
    inputClose: InputsCloseTag?
    next4: CommentsTag* _
    output: OutputsTag* _
    next5: CommentsTag* _
    outputClose: OutputsCloseTag * _
    portMap: PortMapTag* _
    next11: CommentsTag* _
    portMapClose: PortMapCloseTag * _
    backEdges: BackEdgesTag* _
    next12: CommentsTag* _
    backEdgesClose: BackEdgesCloseTag * _
    body: BodyTag* _
    next13: CommentsTag* _
    bodyClose: BodyCloseTag * _
    next6: CommentsTag* _
    weights: WeightsTag?
    next7: CommentsTag* _
    biases: BiasesTag?
    // biases can be before weights
    next8: CommentsTag* _
    weights2: WeightsTag?
    next9: CommentsTag* _
    blobs: BlobsTag?
    next10: CommentsTag* _
    rest: LayerCloseTag {
        var key = '';
        if (output){
            key = Object.keys(output)[0];
            start[key] = output[key];
        }

        if (input){
            key = Object.keys(input)[0];
            start[key] = input[key];
        }

        if (data) {
            key = Object.keys(data)[0];
            start[key] = data[key];
        }

        if (weights) {
            key = Object.keys(weights)[0];
            start[key] = weights[key];
        }

        if (weights2) {
            key = Object.keys(weights2)[0];
            start[key] = weights2[key];
        }

        if (biases) {
            key = Object.keys(biases)[0];
            start[key] = biases[key];
        }
        
        if (blobs) {
            var keys = Object.keys(blobs);
            keys.reduce(function(el1, el2) {
              el1[el2] = blobs[el2];
              return el1;
            }, start);
        }
        
        if (body) {
        	start.nestedIR = body;
        }
        
        if (portMap && portMap[0]) {
        	start.mappingForNestedIR = portMap[0].output[0];
        }
        
        if (backEdges && backEdges[0] && backEdges[0][1]) {
        	start.backEdgesForNestedIR = backEdges[0][1];
        }

        return start;
    }

LayerCloseTag
    = ("</layer>")

LayerStartTag
    = 
    next1: CommentsTag* _
    first:("<layer")
     next2: CommentsTag* _
     rest: tag_attr_pair* ">" _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }

// ----------------------------------LayerTag <<<<<<<<<<<<<<<<<<<<<<<<<<<

OutputsTag
    = 
    next1: CommentsTag* _
    first:(OutputsStartTag _ )
    next2: CommentsTag* _
    rest: (_ input:OutputTag {return input;})* {
        return {'output': rest};
    }

OutputsCloseTag
    = ("</output>") _

OutputTag
    = 
    next1: CommentsTag* _
    port: PortTag _
    next2: CommentsTag* _
    closePort: PortCloseTag _  {
        return {
                  'id':port[2].id,
                  'dims': port[6]
               };
    }

OutputsStartTag
    = first:("<output>")

// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

InputsTag
    = 
    next1: CommentsTag* _
    first:(InputsStartTag _ )
    next2: CommentsTag* _
    rest: (_ input:InputTag {return input;})* {
        return {'input': rest};
    }

InputsCloseTag
    = ("</input>") _

InputTag
    = 
    next1: CommentsTag* _
    port: PortTag _
    next2: CommentsTag* _
    closePort: PortCloseTag _  {
        return {
                  'id':port[2].id,
                  'dims': port[6]
               };
    }

InputsStartTag
    = first:("<input>")
    
// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

PortTag
    = 
    next1: CommentsTag* _
    start: PortStartTag _
    next2: CommentsTag* _
    rest: (_ dim:DimTag {return dim;})*

PortStartTag
    = 
    next1: CommentsTag* _
    first:("<port")
    next2: CommentsTag* _
    rest: tag_attr_pair* ">" _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }

PortCloseTag
    = first:("</port>")
// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

PortMapTag
    = 
    next1: CommentsTag* _
    first:(PortMapStartTag _ )
    next2: CommentsTag* _
    rest: (_ input:PortMapContainerTag {return input;})* {
        return {'output': rest};
    }

PortMapCloseTag
    = ("</port_map>") _

PortMapContainerTag
    = 
    next1: CommentsTag* _
    inputs: (PortMapInputTag _)*
    outputs: (PortMapOutputTag _)*
    next2: CommentsTag* _
    closePort: PortMapCloseTag _  {
        var inputValues = inputs.map(function(el){
        	return el[0];
        });
        var outputValues = outputs.map(function(el){
        	return el[0];
        });
        return {input: inputValues, output: outputValues};
    }

PortMapStartTag
    = first:("<port_map>")


// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

DimTag
    = 
    start: DimStartTag _
    next2: CommentsTag* _
    value: tag_value _
    next3: CommentsTag* _
    rest: DimCloseTag _ {
        return value;
    }

DimStartTag
    = 
    next1: CommentsTag* _
    first:("<dim>")

DimCloseTag
    = 
    next1: CommentsTag* _
    first:("</dim>")

// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<
// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<    
DataTag
    = 
    next1: CommentsTag* _
    start: DataStartTag _ "/>" _{
        return {'data': start};
    }

DataStartTag
    = 
    next1: CommentsTag* _
    first:("<"  ([a-z]* [\_\-]{1})? ([a-z]* [\_\-]{1})? "data")
    rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }
// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

BlobsTag
    = 
    next1: CommentsTag* _
    start: BlobsStartTag _
    next2: CommentsTag* _
    weights: WeightsTag? _
    next3: CommentsTag* _
    biases: BiasesTag? _
    // biases can be before weights
    next4: CommentsTag* _
    weights2: WeightsTag? _
    next5: CommentsTag* _
    customs: CustomBlobsTag* _
    next6: CommentsTag* _
    end: BlobsEndTag _ {
        var key = '';
        var res = {};
        if (weights) {
            key = Object.keys(weights)[0];
            res[key] = weights[key];
        }

        if (weights2) {
            key = Object.keys(weights2)[0];
            res[key] = weights2[key];
        }

        if (biases) {
            key = Object.keys(biases)[0];
            res[key] = biases[key];
        }
        
        if (customs) {
            customs.reduce(function(acc, val, key) {
                var customKey = Object.keys(val)[0];
                var newName = customKey + key;
                acc[newName] = val[customKey];
                return acc;
            }, res);
        }

        return res;
    }

BlobsStartTag
    = 
    next1: CommentsTag* _
    first:("<blobs>")

BlobsEndTag
    = 
    next1: CommentsTag* _
    first:("</blobs>")

// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

CustomBlobsTag
    = start: CustomBlobsStartTag _ "/>" _{
        return {'custom': start};
    }

CustomBlobsStartTag
    = first:("<custom")
    rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }

// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<


// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

WeightsTag
    = start: WeightsStartTag _ "/>" _{
        return {'weights': start};
    }

WeightsStartTag
    = first:("<weights")
    rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }

// ----------------------------------Layer <<<<<<<<<<<<<<<<<<<<<<<<<<<

BiasesTag
    = start: BiasesStartTag _ "/>" _{
        return  {'biases': start};
    }

BiasesStartTag
    = first:("<biases")
    rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }
// ------------------------------------ Edges >>>>>>>>>>>>>>>>>>>>>>>>

EdgesTag
    = first:("<edges>" _ )
    rest: (_ layer:EdgeTag {return layer;})*

EdgesCloseTag
    = ("</edges>")

// ------------------------------------ 

BackEdgesTag
    = first:("<back_edges>" _ )
    rest: (_ layer:EdgeTag {return layer;})*

BackEdgesCloseTag
    = ("</back_edges>")
    


// ------------------------------------ 

BodyTag
    = first:("<body>" _ )
    next2: CommentsTag* _
    layers: LayersTag _
    next3: CommentsTag* _
    close: LayersCloseTag _
    next4: CommentsTag* _
    edges: EdgesTag _
    next5: CommentsTag* _
    closeEdges: EdgesCloseTag _
    next6: CommentsTag* _
    rest: _ {
    	console.log({layers: layers, edges: edges[1]});
    	return {layers: layers, edges: edges[1]};
    }

BodyCloseTag
    = ("</body>")

// ------------------------------------ PortMapInputTag >>>>>>>>>>>>>>

PortMapInputTag
    = start: PortMapInputStartTag _ "/>" {
        return start;
    }

PortMapInputStartTag
    = first:("<input")
     rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }
    
// ------------------------------------ PortMapInputTag <<<<<<<<<<<<<<

// ------------------------------------ PortMapOutputTag >>>>>>>>>>>>>>

PortMapOutputTag
    = start: PortMapOutputStartTag _ "/>" {
        return start;
    }

PortMapOutputStartTag
    = first:("<output")
     rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value;
            return el1;
        }, {});
    }
    
// ------------------------------------ PortMapOutputTag <<<<<<<<<<<<<<

// ------------------------------------ Edge >>>>>>>>>>>>>>>>>>>>>>>>>

EdgeTag
    = start: EdgeStartTag _ "/>" {
        return start;
    }

EdgeCloseTag
    = ("</edge>")

EdgeStartTag
    = first:("<edge")
     rest: tag_attr_pair* _ {
        return rest.reduce(function(el1, el2) {
            el1[el2.key] = el2.value.replace(/"/g, '');
            return el1;
        }, {});
    }

CloseTag
    = "</" tag_name tag_attr_pair* ">" _

any
    = [ =/A-Za-z"0-9 \<\-\>]*

tag_name
    = head:([A-Za-z]+){
        return head.join('');
    }

tag_attr_pair
    = _ name:attr_name '=' value:attr_value _ {
        return {key: name, value :value}
    }

attr_value
    = head:(_'"' [A-Za-z0-9/\.\-\_\, ]* '"'_){
        return head.toString().split(',').join('').slice(1, -1);
    }

attr_name
    = head:([0-9A-Za-z\-\_]+) {
        return head.join('');
    }

tag_value
    = head:([A-Za-z0-9]*){
        return head.toString().split(',').join('');
    }

_ "whitespace"
  = [ \t\n\r]*
