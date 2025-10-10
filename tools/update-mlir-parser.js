#!/usr/bin/env node

/**
 * Script to download and update mlir-js-parser dependencies
 * Usage: node tools/update-mlir-parser.js [version]
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MLIR_PARSER_REPO = 'tucan9389/mlir-js-parser';
const DEFAULT_VERSION = 'v0.1';

async function downloadFile(url, destPath) {
    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(destPath);
        
        const request = https.get(url, (response) => {
            // Handle redirects
            if (response.statusCode === 302 || response.statusCode === 301) {
                const redirectUrl = response.headers.location;
                console.log(`Following redirect...`);
                file.close();
                try { fs.unlinkSync(destPath); } catch {} // Clean up partial file
                downloadFile(redirectUrl, destPath).then(resolve).catch(reject);
                return;
            }
            
            if (response.statusCode !== 200) {
                reject(new Error(`Failed to download ${url}: ${response.statusCode}`));
                return;
            }
            
            response.pipe(file);
            file.on('finish', () => {
                file.close();
                resolve();
            });
        });
        
        request.on('error', (error) => {
            file.close();
            try { fs.unlinkSync(destPath); } catch {} // Clean up partial file
            reject(error);
        });
    });
}

async function updateMlirParser(version = DEFAULT_VERSION) {
    const baseUrl = `https://github.com/${MLIR_PARSER_REPO}/releases/download/${version}/`;
    const sourceDir = path.join(__dirname, '..', 'source');
    
    const files = [
        'mlir_parser.js',
        'mlir_parser.wasm',
        // Save bindings from upstream as mlir-bindings.js locally to avoid collisions
        ['bindings.js', 'mlir-bindings.js']
    ];
    
    console.log(`Updating mlir-js-parser to version ${version}...`);
    console.log(`Downloading from: ${baseUrl}`);
    
    for (const entry of files) {
        const [remote, local] = Array.isArray(entry) ? entry : [entry, entry];
        const url = baseUrl + remote;
        const destPath = path.join(sourceDir, local);
        
        console.log(`Downloading ${local}...`);
        try {
            await downloadFile(url, destPath);
            console.log(`✓ Downloaded ${local}`);
        } catch (error) {
            console.error(`✗ Failed to download ${local}: ${error.message}`);
            console.error(`URL: ${url}`);
            process.exit(1);
        }
    }
    
    console.log('✓ All files updated successfully');
    console.log('You can now use MLIR files locally without network access.');
}

const version = process.argv[2] || DEFAULT_VERSION;
updateMlirParser(version).catch(console.error);