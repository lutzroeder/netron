/**
 * OnnxSlim integration module for Netron
 *
 * This module provides UI and API for modifying ONNX models using OnnxSlim.
 */

const onnxslim = {};

/**
 * Convert Uint8Array to base64 string
 */
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * OnnxSlim API client
 */
onnxslim.Client = class {

    constructor(baseUrl = '', useElectron = false) {
        this._baseUrl = baseUrl;
        this._useElectron = useElectron;
    }

    /**
     * Get available OnnxSlim operations
     */
    async getOperations() {
        return this._request({ operation: 'get_operations' });
    }

    /**
     * Optimize an ONNX model
     */
    async slim(inputPath, options = {}) {
        return this._request({
            operation: 'slim',
            input_path: inputPath,
            output_path: options.outputPath,
            no_shape_infer: options.noShapeInfer || false,
            skip_optimizations: options.skipOptimizations,
            size_threshold: options.sizeThreshold,
            model_data_base64: options.modelDataBase64,
            model_name: options.modelName
        });
    }

    /**
     * Convert model data type
     */
    async convertDtype(inputPath, dtype, options = {}) {
        return this._request({
            operation: 'convert_dtype',
            input_path: inputPath,
            output_path: options.outputPath,
            dtype: dtype,
            model_data_base64: options.modelDataBase64,
            model_name: options.modelName
        });
    }

    /**
     * Modify model inputs
     */
    async modifyInputs(inputPath, options = {}) {
        return this._request({
            operation: 'modify_inputs',
            input_path: inputPath,
            output_path: options.outputPath,
            inputs: options.inputs,
            input_shapes: options.inputShapes,
            model_data_base64: options.modelDataBase64,
            model_name: options.modelName
        });
    }

    /**
     * Modify model outputs
     */
    async modifyOutputs(inputPath, outputs, options = {}) {
        return this._request({
            operation: 'modify_outputs',
            input_path: inputPath,
            output_path: options.outputPath,
            outputs: outputs,
            model_data_base64: options.modelDataBase64,
            model_name: options.modelName
        });
    }

    /**
     * Inspect model without modifying
     */
    async inspect(inputPath, options = {}) {
        return this._request({
            operation: 'inspect',
            input_path: inputPath,
            model_data_base64: options.modelDataBase64,
            model_name: options.modelName
        });
    }

    async _request(data) {
        if (this._useElectron && typeof window !== 'undefined' && window.require) {
            // Use Electron IPC
            return this._electronRequest(data);
        } else {
            // Use HTTP API
            return this._httpRequest(data);
        }
    }

    async _httpRequest(data) {
        const response = await fetch(`${this._baseUrl}/api/onnxslim`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        return response.json();
    }

    async _electronRequest(data) {
        return new Promise((resolve, reject) => {
            const electron = window.require('electron');
            electron.ipcRenderer.once('onnxslim-complete', (event, result) => {
                if (result.error) {
                    resolve({ status: 'error', error: result.error });
                } else {
                    resolve(result.value);
                }
            });
            electron.ipcRenderer.send('onnxslim', data);
        });
    }
};

/**
 * OnnxSlim Sidebar for model modification UI
 */
onnxslim.Sidebar = class {

    constructor(view, model, modelPath, modelData = null) {
        this._view = view;
        this._model = model;
        this._modelPath = modelPath;
        // Store raw model bytes for in-memory optimization
        this._modelData = modelData;
        this._modelDataBase64 = null;
        // Convert to base64 if we have model data
        if (modelData && modelData.length > 0) {
            try {
                this._modelDataBase64 = arrayBufferToBase64(modelData);
            } catch {
                // Ignore errors converting to base64
            }
        }
        // Determine if we're running in Electron
        const isElectron = view.host && view.host.type === 'Electron';
        this._client = new onnxslim.Client('', isElectron);
        this._events = {};
    }

    get identifier() {
        return 'onnxslim';
    }

    on(event, callback) {
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        if (this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    render() {
        const document = this._view.host.document;

        this.element = document.createElement('div');
        this.element.className = 'sidebar-onnxslim';

        // Header
        const header = document.createElement('div');
        header.className = 'sidebar-header';
        header.innerHTML = '<h2>OnnxSlim Model Tools</h2>';
        this.element.appendChild(header);

        // Info section
        const info = document.createElement('div');
        info.className = 'sidebar-section';
        const inMemory = this._modelDataBase64 ? ' (in-memory)' : '';
        info.innerHTML = `
            <div class="sidebar-item-value">
                <p>Optimize and modify your ONNX model using OnnxSlim.</p>
                <p><small>Model: ${this._modelPath || 'Unknown'}${inMemory}</small></p>
            </div>
        `;
        this.element.appendChild(info);

        // Optimize section
        this._addSection('Optimize Model', () => this._createOptimizeSection());

        // Convert dtype section
        this._addSection('Convert Data Type', () => this._createDtypeSection());

        // Modify inputs section
        this._addSection('Modify Inputs', () => this._createInputsSection());

        // Status area
        this._statusArea = document.createElement('div');
        this._statusArea.className = 'sidebar-section sidebar-status';
        this.element.appendChild(this._statusArea);

        return this.element;
    }

    _addSection(title, contentCreator) {
        const document = this._view.host.document;
        const section = document.createElement('div');
        section.className = 'sidebar-section';

        const sectionHeader = document.createElement('div');
        sectionHeader.className = 'sidebar-section-header';
        sectionHeader.innerHTML = `<span>${title}</span>`;
        section.appendChild(sectionHeader);

        const content = contentCreator();
        section.appendChild(content);

        this.element.appendChild(section);
    }

    _createOptimizeSection() {
        const document = this._view.host.document;
        const container = document.createElement('div');
        container.className = 'sidebar-section-content';

        // Options
        const options = document.createElement('div');
        options.className = 'sidebar-item';

        // Shape inference checkbox
        const shapeInferLabel = document.createElement('label');
        shapeInferLabel.className = 'sidebar-checkbox';
        const shapeInferCheck = document.createElement('input');
        shapeInferCheck.type = 'checkbox';
        shapeInferCheck.id = 'onnxslim-shape-infer';
        shapeInferLabel.appendChild(shapeInferCheck);
        shapeInferLabel.appendChild(document.createTextNode(' Enable shape inference'));
        options.appendChild(shapeInferLabel);
        options.appendChild(document.createElement('br'));

        // Skip optimizations
        const skipLabel = document.createElement('div');
        skipLabel.className = 'sidebar-item-label';
        skipLabel.textContent = 'Skip optimizations:';
        options.appendChild(skipLabel);

        const optimizations = [
            { id: 'constant_folding', label: 'Constant Folding' },
            { id: 'graph_fusion', label: 'Graph Fusion' },
            { id: 'dead_node_elimination', label: 'Dead Node Elimination' },
            { id: 'subexpression_elimination', label: 'Subexpression Elimination' },
            { id: 'weight_tying', label: 'Weight Tying' }
        ];

        for (const opt of optimizations) {
            const label = document.createElement('label');
            label.className = 'sidebar-checkbox';
            const check = document.createElement('input');
            check.type = 'checkbox';
            check.id = `onnxslim-skip-${opt.id}`;
            check.dataset.optimization = opt.id;
            label.appendChild(check);
            label.appendChild(document.createTextNode(` ${opt.label}`));
            options.appendChild(label);
            options.appendChild(document.createElement('br'));
        }

        container.appendChild(options);

        // Optimize button
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'sidebar-button-container';
        const button = document.createElement('button');
        button.className = 'sidebar-button';
        button.textContent = 'Optimize Model';
        button.addEventListener('click', () => this._handleOptimize());
        buttonContainer.appendChild(button);
        container.appendChild(buttonContainer);

        return container;
    }

    _createDtypeSection() {
        const document = this._view.host.document;
        const container = document.createElement('div');
        container.className = 'sidebar-section-content';

        const options = document.createElement('div');
        options.className = 'sidebar-item';

        // Dtype select
        const selectLabel = document.createElement('label');
        selectLabel.textContent = 'Target data type: ';
        const select = document.createElement('select');
        select.id = 'onnxslim-dtype';
        select.innerHTML = `
            <option value="fp16">Float16 (fp16)</option>
            <option value="fp32">Float32 (fp32)</option>
        `;
        selectLabel.appendChild(select);
        options.appendChild(selectLabel);
        container.appendChild(options);

        // Convert button
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'sidebar-button-container';
        const button = document.createElement('button');
        button.className = 'sidebar-button';
        button.textContent = 'Convert Data Type';
        button.addEventListener('click', () => this._handleConvertDtype());
        buttonContainer.appendChild(button);
        container.appendChild(buttonContainer);

        return container;
    }

    _createInputsSection() {
        const document = this._view.host.document;
        const container = document.createElement('div');
        container.className = 'sidebar-section-content';

        const options = document.createElement('div');
        options.className = 'sidebar-item';

        // Input shapes
        const shapesLabel = document.createElement('div');
        shapesLabel.className = 'sidebar-item-label';
        shapesLabel.textContent = 'Input shapes (name:dim1,dim2,...):';
        options.appendChild(shapesLabel);

        const shapesInput = document.createElement('input');
        shapesInput.type = 'text';
        shapesInput.id = 'onnxslim-input-shapes';
        shapesInput.className = 'sidebar-input';
        shapesInput.placeholder = 'e.g., input:1,3,224,224';
        options.appendChild(shapesInput);
        options.appendChild(document.createElement('br'));

        // Input dtypes
        const dtypesLabel = document.createElement('div');
        dtypesLabel.className = 'sidebar-item-label';
        dtypesLabel.textContent = 'Input types (name:dtype):';
        options.appendChild(dtypesLabel);

        const dtypesInput = document.createElement('input');
        dtypesInput.type = 'text';
        dtypesInput.id = 'onnxslim-inputs';
        dtypesInput.className = 'sidebar-input';
        dtypesInput.placeholder = 'e.g., input:fp32';
        options.appendChild(dtypesInput);

        container.appendChild(options);

        // Modify button
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'sidebar-button-container';
        const button = document.createElement('button');
        button.className = 'sidebar-button';
        button.textContent = 'Modify Inputs';
        button.addEventListener('click', () => this._handleModifyInputs());
        buttonContainer.appendChild(button);
        container.appendChild(buttonContainer);

        return container;
    }

    async _handleOptimize() {
        // Check if we have either a path or model data
        if (!this._modelPath && !this._modelDataBase64) {
            this._showStatus('error', 'No model path or data available');
            return;
        }

        this._showStatus('info', 'Optimizing model...');

        const document = this._view.host.document;
        const noShapeInfer = !document.getElementById('onnxslim-shape-infer').checked;

        const skipOptimizations = [];
        const checkboxes = document.querySelectorAll('[id^="onnxslim-skip-"]');
        for (const checkbox of checkboxes) {
            if (checkbox.checked && checkbox.dataset.optimization) {
                skipOptimizations.push(checkbox.dataset.optimization);
            }
        }

        try {
            const result = await this._client.slim(this._modelPath, {
                noShapeInfer,
                skipOptimizations: skipOptimizations.length > 0 ? skipOptimizations : undefined,
                modelDataBase64: this._modelDataBase64,
                modelName: this._modelPath || 'model.onnx'
            });

            if (result.status === 'success') {
                this._showStatus('success', `Model optimized successfully!\nSaved to: ${result.output_path}`, result.output_path);
            } else {
                this._showStatus('error', `Optimization failed: ${result.error}`);
            }
        } catch (error) {
            this._showStatus('error', `Error: ${error.message}`);
        }
    }

    async _handleConvertDtype() {
        // Check if we have either a path or model data
        if (!this._modelPath && !this._modelDataBase64) {
            this._showStatus('error', 'No model path or data available');
            return;
        }

        const document = this._view.host.document;
        const dtype = document.getElementById('onnxslim-dtype').value;

        this._showStatus('info', `Converting to ${dtype}...`);

        try {
            const result = await this._client.convertDtype(this._modelPath, dtype, {
                modelDataBase64: this._modelDataBase64,
                modelName: this._modelPath || 'model.onnx'
            });

            if (result.status === 'success') {
                this._showStatus('success', `Model converted to ${dtype}!\nSaved to: ${result.output_path}`, result.output_path);
            } else {
                this._showStatus('error', `Conversion failed: ${result.error}`);
            }
        } catch (error) {
            this._showStatus('error', `Error: ${error.message}`);
        }
    }

    async _handleModifyInputs() {
        // Check if we have either a path or model data
        if (!this._modelPath && !this._modelDataBase64) {
            this._showStatus('error', 'No model path or data available');
            return;
        }

        const document = this._view.host.document;
        const shapesInput = document.getElementById('onnxslim-input-shapes').value.trim();
        const inputsInput = document.getElementById('onnxslim-inputs').value.trim();

        if (!shapesInput && !inputsInput) {
            this._showStatus('error', 'Please specify input shapes or types to modify');
            return;
        }

        const inputShapes = shapesInput ? shapesInput.split(/\s*,\s*(?=[a-zA-Z])/).filter(s => s) : undefined;
        const inputs = inputsInput ? inputsInput.split(/\s*,\s*(?=[a-zA-Z])/).filter(s => s) : undefined;

        this._showStatus('info', 'Modifying inputs...');

        try {
            const result = await this._client.modifyInputs(this._modelPath, {
                inputShapes,
                inputs,
                modelDataBase64: this._modelDataBase64,
                modelName: this._modelPath || 'model.onnx'
            });

            if (result.status === 'success') {
                this._showStatus('success', `Inputs modified!\nSaved to: ${result.output_path}`, result.output_path);
            } else {
                this._showStatus('error', `Modification failed: ${result.error}`);
            }
        } catch (error) {
            this._showStatus('error', `Error: ${error.message}`);
        }
    }

    _showStatus(type, message, outputPath = null) {
        if (!this._statusArea) {
            return;
        }
        const document = this._view.host.document;
        const classes = {
            info: 'sidebar-status-info',
            success: 'sidebar-status-success',
            error: 'sidebar-status-error'
        };
        this._statusArea.className = `sidebar-section sidebar-status ${classes[type] || ''}`;
        this._statusArea.innerHTML = `<pre>${message}</pre>`;

        // Add action buttons for successful operations with output path
        if (type === 'success' && outputPath) {
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'sidebar-button-container';
            buttonContainer.style.marginTop = '10px';
            buttonContainer.style.display = 'flex';
            buttonContainer.style.gap = '20px';

            const openBtn = document.createElement('button');
            openBtn.className = 'sidebar-button';
            openBtn.textContent = 'Open Model';
            openBtn.addEventListener('click', () => {
                this.emit('model-modified', { path: outputPath, operation: 'open' });
            });
            buttonContainer.appendChild(openBtn);

            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'sidebar-button';
            downloadBtn.textContent = 'Download';
            downloadBtn.addEventListener('click', () => {
                if (this._view.host && this._view.host.type === 'Electron') {
                    // In Electron mode, emit event to trigger save dialog
                    this.emit('model-modified', { path: outputPath, operation: 'download' });
                } else {
                    // In browser mode, use the API endpoint
                    window.location.href = `/api/download?path=${encodeURIComponent(outputPath)}`;
                }
            });
            buttonContainer.appendChild(downloadBtn);

            this._statusArea.appendChild(buttonContainer);
        }
    }
};

/**
 * Check if OnnxSlim is available
 */
onnxslim.isAvailable = async function(baseUrl = '') {
    try {
        const client = new onnxslim.Client(baseUrl);
        const result = await client.getOperations();
        return result.status === 'success';
    } catch {
        return false;
    }
};

export const Client = onnxslim.Client;
export const Sidebar = onnxslim.Sidebar;
export const isAvailable = onnxslim.isAvailable;
