
/* eslint-env es2017 */

const child_process = require('child_process');
const crypto = require('crypto');
const http = require('http');
const https = require('https');
const fs = require('fs');
const os = require('os');
const path = require('path');

let configuration = null;

const load = () => {
    const file = path.join(__dirname, 'package.json');
    const content = fs.readFileSync(file, 'utf-8');
    configuration = JSON.parse(content);
};

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message + os.EOL);
    }
};

const rm = (...args) => {
    write('rm ' + path.join(...args));
    const dir = path.join(__dirname, ...args);
    fs.rmSync(dir, { recursive: true, force: true });
};

const mkdir = (...args) => {
    write('mkdir ' + path.join(...args));
    const dir = path.join(__dirname, ...args);
    fs.mkdirSync(dir, { recursive: true });
};

const exec = (command) => {
    try {
        child_process.execSync(command, { cwd: __dirname, stdio: [ 0,1,2 ] });
    } catch (error) {
        process.exit(1);
    }
};

const sleep = (delay) => {
    return new Promise((resolve) => {
        setTimeout(resolve, delay);
    });
};

const install = () => {
    const node_modules = path.join(__dirname, 'node_modules');
    if (!fs.existsSync(node_modules)) {
        const options = { cwd: __dirname, stdio: [ 0,1,2 ] };
        child_process.execSync('npm install', options);
    }
};

const clean = () => {
    rm('dist');
    rm('node_modules');
    rm('package-lock.json');
};

const reset = () => {
    clean();
    rm('third_party', 'env');
    rm('third_party', 'source');
};

const build = async (target) => {
    switch (target || read()) {
        case 'web': {
            write('build web');
            rm('dist', 'web');
            mkdir('dist', 'web');
            write('cp source/dir dist/dir');
            const source_dir = path.join(__dirname, 'source');
            const dist_dir = path.join(__dirname, 'dist', 'web');
            const extensions = new Set([ 'html', 'css', 'js', 'json', 'ico', 'png' ]);
            for (const file of fs.readdirSync(source_dir)) {
                if (extensions.has(file.split('.').pop())) {
                    fs.copyFileSync(path.join(source_dir, file), path.join(dist_dir, file));
                }
            }
            rm('dist', 'web', 'app.js');
            rm('dist', 'web', 'electron.js');
            const manifestFile = path.join(__dirname, 'package.json');
            const contentFile = path.join(__dirname, 'dist', 'web', 'index.html');
            const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf-8'));
            let content = fs.readFileSync(contentFile, 'utf-8');
            content = content.replace(/(<meta\s*name="version"\s*content=")(.*)(">)/m, (match, p1, p2, p3) => {
                return p1 + manifest.version + p3;
            });
            content = content.replace(/(<meta\s*name="date"\s*content=")(.*)(">)/m, (match, p1, p2, p3) => {
                return p1 + manifest.date + p3;
            });
            fs.writeFileSync(contentFile, content, 'utf-8');
            break;
        }
        case 'electron': {
            write('build electron');
            install();
            exec('npx electron-builder install-app-deps');
            exec('npx electron-builder install-app-deps');
            exec('npx electron-builder --mac --universal --publish never -c.mac.identity=null');
            exec('npx electron-builder --win --x64 --arm64 --publish never');
            exec('npx electron-builder --linux appimage --x64 --publish never');
            exec('npx electron-builder --linux snap --x64 --publish never');
            break;
        }
        case 'python': {
            write('build python');
            exec('python package.py build version');
            exec('python -m pip install --user build wheel --quiet');
            exec('python -m build --no-isolation --wheel --outdir dist/pypi dist/pypi');
            if (read('install')) {
                exec('python -m pip install --force-reinstall dist/pypi/*.whl');
            }
            break;
        }
        default: {
            write('build');
            rm('dist');
            install();
            await build('web');
            await build('electron');
            await build('python');
            break;
        }
    }
};

const publish = async (target) => {
    const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
    const GITHUB_USER = process.env.GITHUB_USER;
    switch (target || read()) {
        case 'web': {
            write('publish web');
            build('web');
            rm('dist', 'gh-pages');
            const url = 'https://x-access-token:' + GITHUB_TOKEN + '@github.com/' + GITHUB_USER + '/netron.git';
            exec('git clone --depth=1 ' + url + ' --branch gh-pages ./dist/gh-pages 2>&1 > /dev/null');
            write('cp dist/web dist/gh-pages');
            const source_dir = path.join(__dirname, 'dist', 'web');
            const target_dir = path.join(__dirname, 'dist', 'gh-pages');
            for (const file of fs.readdirSync(target_dir).filter((file) => file !== '.git')) {
                fs.unlinkSync(path.join(target_dir, file));
            }
            for (const file of fs.readdirSync(source_dir)) {
                fs.copyFileSync(path.join(source_dir, file), path.join(target_dir, file));
            }
            exec('git -C dist/gh-pages add --all');
            exec('git -C dist/gh-pages commit --amend --no-edit');
            exec('git -C dist/gh-pages push --force origin gh-pages');
            break;
        }
        case 'electron': {
            write('publish electron');
            install();
            exec('npx electron-builder install-app-deps');
            exec('npx electron-builder --mac --universal --publish always');
            exec('npx electron-builder --win --x64 --arm64 --publish always');
            exec('npx electron-builder --linux appimage --x64 --publish always');
            exec('npx electron-builder --linux snap --x64 --publish always');
            break;
        }
        case 'python': {
            write('publish python');
            build('python');
            exec('python -m pip install --user twine');
            exec('python -m twine upload --non-interactive --skip-existing --verbose dist/pypi/*.whl');
            break;
        }
        case 'cask': {
            write('publish cask');
            const authorization = 'Authorization: token ' + GITHUB_TOKEN;
            exec('curl -s -H "' + authorization + '" -X "DELETE" https://api.github.com/repos/' + GITHUB_USER + '/homebrew-cask 2>&1 > /dev/null');
            await sleep(4000);
            exec('curl -s -H "' + authorization + '"' + " https://api.github.com/repos/Homebrew/homebrew-cask/forks -d '' 2>&1 > /dev/null");
            rm('dist', 'homebrew-cask');
            exec('git clone --depth=2 https://x-access-token:' + GITHUB_TOKEN + '@github.com/' + GITHUB_USER + '/homebrew-cask.git ./dist/homebrew-cask');
            const repository = 'https://github.com/' + configuration.repository;
            const url = repository + '/releases/download/v#{version}/' + configuration.productName + '-#{version}-mac.zip';
            const location = url.replace(/#{version}/g, configuration.version);
            const sha256 = crypto.createHash('sha256').update(await get(location)).digest('hex').toLowerCase();
            const paths = [ 'dist', 'homebrew-cask' ];
            mkdir(...paths);
            const file = path.join(__dirname, ...paths, 'netron.rb');
            fs.writeFileSync(file, [
                'cask "' + configuration.name + '" do',
                '  version "' + configuration.version + '"',
                '  sha256 "' + sha256 + '"',
                '',
                '  url "' + url + '"',
                '  name "' + configuration.productName + '"',
                '  desc "' + configuration.description + '"',
                '  homepage "' + repository + '"',
                '',
                '  auto_updates true',
                '',
                '  app "' + configuration.productName + '.app"',
                'end',
                ''
            ].join('\n'));
            exec('git -C dist/homebrew-cask add --all');
            exec('git -C dist/homebrew-cask commit -m "Update ' + configuration.name + ' to ' + configuration.version + '"');
            exec('git -C dist/homebrew-cask push');
            exec('curl -H "' + authorization + '"' + ' https://api.github.com/repos/Homebrew/homebrew-cask/pulls -d "{\\"title\\":\\"Update ' + configuration.name + ' to ' + configuration.version + '\\",\\"base\\":\\"master\\",\\"head\\":\\"' + GITHUB_USER + ':master\\",\\"body\\":\\"Update version and sha256.\\"}" 2>&1 > /dev/null');
            rm('dist', 'homebrew-cask');
            break;
        }
        case 'winget': {
            write('publish winget');
            const authorization = 'Authorization: token ' + GITHUB_TOKEN;
            write('delete github winget-pkgs');
            exec('curl -s -H "' + authorization + '" -X "DELETE" https://api.github.com/repos/' + GITHUB_USER + '/winget-pkgs');
            await sleep(4000);
            write('create github winget-pkgs');
            exec('curl -s -H "' + authorization + '"' + " https://api.github.com/repos/microsoft/winget-pkgs/forks -d ''");
            rm('dist', 'winget-pkgs');
            await sleep(4000);
            write('clone github winget-pkgs');
            exec('git clone --depth=2 https://x-access-token:' + GITHUB_TOKEN + '@github.com/' + GITHUB_USER + '/winget-pkgs.git dist/winget-pkgs');
            const name = configuration.name;
            const version = configuration.version;
            const productName = configuration.productName;
            const publisher = configuration.author.name;
            const packageIdentifier = publisher.replace(' ', '') + '.' + productName;
            const copyright = 'Copyright (c) ' + publisher;
            const repository = 'https://github.com/' + configuration.repository;
            const url = repository + '/releases/download/v' + version + '/' + productName + '-Setup-' + version + '.exe';
            const extensions = configuration.build.fileAssociations.map((entry) => '- ' + entry.ext).sort().join('\n');
            write('download ' + url);
            const sha256 = crypto.createHash('sha256').update(await get(url)).digest('hex').toUpperCase();
            const paths = [ 'dist', 'winget-pkgs', 'manifests', publisher[0].toLowerCase(), publisher.replace(' ', ''), productName ];
            // rm(...paths);
            // exec('git -C dist/winget-pkgs add --all');
            // exec('git -C dist/winget-pkgs commit -m "Remove ' + configuration.name + '"');
            paths.push(version);
            mkdir(...paths);
            write('create manifest');
            const manifestFile = path.join(__dirname, ...paths, packageIdentifier);
            fs.writeFileSync(manifestFile + '.yaml', [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.version.1.2.0.schema.json',
                'PackageIdentifier: ' + packageIdentifier,
                'PackageVersion: ' + version,
                'DefaultLocale: en-US',
                'ManifestType: version',
                'ManifestVersion: 1.2.0',
                ''
            ].join('\n'));
            fs.writeFileSync(manifestFile + '.installer.yaml', [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.installer.1.2.0.schema.json',
                'PackageIdentifier: ' + packageIdentifier,
                'PackageVersion: ' + version,
                'Platform:',
                '- Windows.Desktop',
                'InstallModes:',
                '- silent',
                '- silentWithProgress',
                'Installers:',
                '- Architecture: x86',
                '  Scope: user',
                '  InstallerType: nullsoft',
                '  InstallerUrl: ' + url,
                '  InstallerSha256: ' + sha256,
                '  InstallerLocale: en-US',
                '  InstallerSwitches:',
                '    Custom: /NORESTART',
                '  UpgradeBehavior: install',
                '- Architecture: arm64',
                '  Scope: user',
                '  InstallerType: nullsoft',
                '  InstallerUrl: ' + url,
                '  InstallerSha256: ' + sha256,
                '  InstallerLocale: en-US',
                '  InstallerSwitches:',
                '    Custom: /NORESTART',
                '  UpgradeBehavior: install',
                'FileExtensions:',
                extensions,
                'ManifestType: installer',
                'ManifestVersion: 1.2.0',
                ''
            ].join('\n'));
            fs.writeFileSync(manifestFile + '.locale.en-US.yaml', [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.defaultLocale.1.2.0.schema.json',
                'PackageIdentifier: ' + packageIdentifier,
                'PackageVersion: ' + version,
                'PackageName: ' + productName,
                'PackageLocale: en-US',
                'PackageUrl: ' + repository,
                'Publisher: ' + publisher,
                'PublisherUrl: ' + repository,
                'PublisherSupportUrl: ' + repository + '/issues',
                'Author: ' + publisher,
                'License: ' + configuration.license,
                'Copyright: ' + copyright,
                'CopyrightUrl: ' + repository + '/blob/main/LICENSE',
                'ShortDescription: ' + configuration.description,
                'Description: ' + configuration.description,
                'Moniker: ' + name,
                'Tags:',
                '- machine-learning',
                '- deep-learning',
                '- neural-network',
                'ManifestType: defaultLocale',
                'ManifestVersion: 1.2.0',
                ''
            ].join('\n'));
            write('commit manifest');
            exec('git -C dist/winget-pkgs add --all');
            exec('git -C dist/winget-pkgs commit -m "Update ' + configuration.name + ' to ' + configuration.version + '"');
            exec('git -C dist/winget-pkgs push');
            exec('curl -H "' + authorization + '" https://api.github.com/repos/microsoft/winget-pkgs/pulls -d "{\\"title\\":\\"Update ' + configuration.productName + ' to ' + configuration.version + '\\",\\"base\\":\\"master\\",\\"head\\":\\"' + GITHUB_USER + ':master\\",\\"body\\":\\"\\"}" 2>&1 > /dev/null');
            rm('dist', 'winget-pkgs');
            break;
        }
        default: {
            write('publish');
            rm('dist');
            install();
            await publish('web');
            await publish('electron');
            await publish('python');
            await publish('cask');
            await publish('winget');
            break;
        }
    }
};

const lint = () => {
    install();
    write('eslint');
    exec('npx eslint source/*.js test/*.js publish/*.js tools/*.js');
    write('pylint');
    exec('python -m pip install --upgrade --quiet pylint');
    exec('python -m pylint -sn --recursive=y source test publish tools');
};

const update = () => {
    const targets = process.argv.length > 3 ? process.argv.slice(3) : [
        'armnn',
        'bigdl',
        'caffe',
        'circle',
        'cntk',
        'coreml',
        'dlc',
        'dnn',
        'mnn',
        'mslite',
        'megengine',
        'nnabla',
        'onnx',
        'om',
        'paddle',
        'pytorch',
        'rknn',
        'sklearn',
        'tf',
        'uff',
        'xmodel'
    ];
    for (const target of targets) {
        exec('tools/' + target + ' sync install schema metadata');
    }
};

const get = (url, timeout) => {
    return new Promise((resolve, reject) => {
        const httpModule = url.split(':').shift() === 'https' ? https : http;
        const request = httpModule.request(url, {}, (response) => {
            if (response.statusCode === 200) {
                const data = [];
                let position = 0;
                response.on('data', (chunk) => {
                    data.push(chunk);
                    position += chunk.length;
                    process.stdout.write('  ' + position + ' bytes\r');
                });
                response.on('err', (err) => {
                    reject(err);
                });
                response.on('end', () => {
                    resolve(Buffer.concat(data));
                });
            } else if (response.statusCode === 302) {
                get(response.headers.location).then((data) => {
                    resolve(data);
                }).catch((err) => {
                    reject(err);
                });
            } else {
                const err = new Error("The web request failed with status code " + response.statusCode + " at '" + url + "'.");
                err.type = 'error';
                err.url = url;
                err.status = response.statusCode;
                reject(err);
            }
        });
        request.on("error", (err) => {
            reject(err);
        });
        if (timeout) {
            request.setTimeout(timeout, () => {
                request.destroy();
                const err = new Error("The web request timed out at '" + url + "'.");
                err.type = 'timeout';
                err.url = url;
                reject(err);
            });
        }
        request.end();
    });
};

const pull = () => {
    exec('git fetch --prune origin "refs/tags/*:refs/tags/*"');
    exec('git pull --prune --rebase');
};

const coverage = () => {
    rm('.nyc_output');
    rm('coverage');
    rm('dist', 'nyc');
    mkdir('dist', 'nyc');
    exec('cp package.json dist/nyc');
    exec('cp -R source dist/nyc');
    exec('nyc instrument --compact false source dist/nyc/source');
    exec('nyc --reporter=lcov --instrument npx electron ./dist/nyc');
};

const analyze = () => {
    if (!fs.existsSync('third_party/tools/codeql')) {
        exec('git clone --depth=1 https://github.com/github/codeql.git third_party/tools/codeql');
    }
    rm('dist', 'codeql');
    mkdir('dist', 'codeql', 'netron');
    exec('cp -r publish source test tools dist/codeql/netron/');
    exec('codeql database create dist/codeql/database --source-root dist/codeql/netron --language=javascript --threads=3');
    exec('codeql database analyze dist/codeql/database ./third_party/tools/codeql/javascript/ql/src/codeql-suites/javascript-security-and-quality.qls --format=csv --output=dist/codeql/results.csv --threads=3');
    exec('cat dist/codeql/results.csv');
};

const version = () => {
    const file = path.join(__dirname, 'package.json');
    let content = fs.readFileSync(file, 'utf-8');
    content = content.replace(/(\s*"version":\s")(\d\.\d\.\d)(",)/m, (match, p1, p2, p3) => {
        const version = Array.from((parseInt(p2.split('.').join(''), 10) + 1).toString()).join('.');
        return p1 + version + p3;
    });
    content = content.replace(/(\s*"date":\s")(.*)(",)/m, (match, p1, p2, p3) => {
        const date = new Date().toISOString().split('.').shift().split('T').join(' ');
        return p1 + date + p3;
    });
    fs.writeFileSync(file, content, 'utf-8');
    load();
    exec('git add package.json');
    exec('git commit -m "Update to ' + configuration.version + '"');
    exec('git tag v' + configuration.version);
    exec('git push');
    exec('git push --tags');
};

const args = process.argv.slice(2);

const read = (match) => {
    if (args.length > 0 || (!match || args[0] === match)) {
        return args.shift();
    }
    return null;
};

const next = async () => {
    try {
        const task = read();
        switch (task) {
            case 'install': clean(); break;
            case 'clean': clean(); break;
            case 'reset': reset(); break;
            case 'build': await build(); break;
            case 'publish': await publish(); break;
            case 'version': version(); break;
            case 'lint': lint(); break;
            case 'update': update(); break;
            case 'pull': pull(); break;
            case 'analyze': analyze(); break;
            case 'coverage': coverage(); break;
            default: throw new Error("Unsupported task '" + task + "'.");
        }
    } catch (err) {
        write(err.message);
    }
};

load();
next();
