
import * as flatbuffers from './flatbuffers.js';
import * as python from './python.js';

const catboost = {};

catboost.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = Array.from(buffer).map((c) => String.fromCharCode(c)).join('');
            if (signature === 'CBM1') {
                return context.set('catboost.flatbuffers');
            }
        }
        const obj = await context.peek('pkl');
        if (obj && obj.__class__ && obj.__class__.__module__) {
            const name = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
            if (name.startsWith('catboost.') || name.startsWith('autogluon.tabular.models.catboost.')) {
                return context.set('catboost.pickle', obj);
            }
        }
        return null;
    }

    async open(context) {
        switch (context.type) {
            case 'catboost.flatbuffers': {
                const stream = context.stream;
                const buffer = stream.peek();
                const execution = new python.Execution();
                const obj = execution.invoke('catboost.CatBoostClassifier', []);
                obj._object.flatbuffers = flatbuffers;
                obj._object.schema = await context.require('./catboost-schema');
                obj.load_model(buffer);
                return new catboost.Model(obj._object);
            }
            case 'catboost.pickle': {
                let obj = context.value;
                const name = obj && obj.__class__ && obj.__class__.__module__ && obj.__class__.__name__ ? `${obj.__class__.__module__}.${obj.__class__.__name__}` : '';
                if (name === 'autogluon.tabular.models.catboost.catboost_model.CatBoostModel') {
                    obj = obj.model;
                }
                obj._object.flatbuffers = flatbuffers;
                obj._object.schema = await context.require('./catboost-schema');
                obj.load_model();
                return new catboost.Model(obj._object);
            }
            default: {
                throw new catboost.Error(`Unsupported CatBoost format '${context.type}'.`);
            }
        }
    }
};

catboost.Model = class {

    constructor(obj) {
        this.format = 'CatBoost';
        this.metadata = [];
        for (const [name, value] of obj._get_info_map()) {
            this.metadata.push({ name, value });
        }
        this.modules = [new catboost.Graph(obj)];
    }
};

catboost.Graph = class {

    constructor(obj) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const features = [];
        for (const feature of obj._get_float_features()) {
            const name = feature.FeatureId || `float_feature_${feature.FlatIndex}`;
            const value = new catboost.Value(name, new catboost.TensorType('float32'));
            features.push(value);
            this.inputs.push(new catboost.Argument(name, [value]));
        }
        for (const feature of obj._get_cat_features()) {
            const name = feature.FeatureId || `cat_feature_${feature.FlatIndex}`;
            const value = new catboost.Value(name, new catboost.TensorType('int32'));
            features.push(value);
            this.inputs.push(new catboost.Argument(name, [value]));
        }
        for (const feature of obj._get_text_features()) {
            const name = feature.FeatureId || `text_feature_${feature.FlatIndex}`;
            const value = new catboost.Value(name, new catboost.TensorType('string'));
            features.push(value);
            this.inputs.push(new catboost.Argument(name, [value]));
        }
        for (const feature of obj._get_embedding_features()) {
            const name = feature.FeatureId || `embedding_feature_${feature.FlatIndex}`;
            const value = new catboost.Value(name, new catboost.TensorType('float32[]'));
            features.push(value);
            this.inputs.push(new catboost.Argument(name, [value]));
        }
        const node = new catboost.Node(obj, features);
        this.nodes.push(node);
    }
};

catboost.Argument = class {

    constructor(name, value) {
        this.name = name;
        this.value = value;
    }
};

catboost.Value = class {

    constructor(name, type) {
        this.name = name;
        this.type = type || null;
    }
};

catboost.TensorType = class {

    constructor(dataType) {
        this.dataType = dataType;
        this.shape = null;
    }

    toString() {
        return this.dataType;
    }
};

catboost.Node = class {

    constructor(obj, features) {
        this.name = '';
        this.type = { name: 'CatBoost' };
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        if (features.length > 0) {
            this.inputs.push(new catboost.Argument('features', features));
        }
        const treeCount = obj._get_tree_count();
        if (treeCount > 0) {
            this.attributes.push(new catboost.Argument('tree_count', treeCount));
        }
        const treeSizes = obj._get_tree_sizes();
        if (treeSizes.length > 0) {
            this.attributes.push(new catboost.Argument('tree_sizes', Array.from(treeSizes)));
        }
        const treeSplits = obj._get_tree_splits();
        if (treeSplits.length > 0) {
            this.attributes.push(new catboost.Argument('tree_splits', Array.from(treeSplits)));
        }
        const leafValues = obj._get_leaf_values();
        if (leafValues.length > 0) {
            this.attributes.push(new catboost.Argument('leaf_values', Array.from(leafValues)));
        }
        const leafWeights = obj._get_leaf_weights();
        if (leafWeights.length > 0) {
            this.attributes.push(new catboost.Argument('leaf_weights', Array.from(leafWeights)));
        }
        const borders = obj._get_borders();
        if (borders.length > 0) {
            this.attributes.push(new catboost.Argument('borders', borders));
        }
        const [scale, bias] = obj._get_scale_and_bias();
        if (scale !== undefined && scale !== 1) {
            this.attributes.push(new catboost.Argument('scale', scale));
        }
        if (bias !== undefined && bias !== 0) {
            this.attributes.push(new catboost.Argument('bias', bias));
        }
    }
};

catboost.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading CatBoost model.';
    }
};

export const ModelFactory = catboost.ModelFactory;
