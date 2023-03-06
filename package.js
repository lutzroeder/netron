
/* eslint-env es2017 */

const child_process = require('child_process');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');

let configuration = null;

const load = () => {
    const file = path.join(__dirname, 'package.json');
    const content = fs.readFileSync(file, 'utf-8');
    configuration = JSON.parse(content);
};

const clearLine = () => {
    if (process.stdout.clearLine) {
        process.stdout.clearLine();
    }
};

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message);
    }
};

const writeLine = (message) => {
    write(message + os.EOL);
};

const rm = (...args) => {
    writeLine('rm ' + path.join(...args));
    const dir = path.join(__dirname, ...args);
    fs.rmSync(dir, { recursive: true, force: true });
};

const mkdir = (...args) => {
    writeLine('mkdir ' + path.join(...args));
    const dir = path.join(__dirname, ...args);
    fs.mkdirSync(dir, { recursive: true });
    return dir;
};

const exec = (command) => {
    child_process.execSync(command, { cwd: __dirname, stdio: [ 0,1,2 ] });
};

const sleep = (delay) => {
    return new Promise((resolve) => {
        setTimeout(resolve, delay);
    });
};

const request = async (url, init, status) => {
    const response = await fetch(url, init);
    if (status !== false && !response.ok) {
        throw new Error(response.status.toString());
    }
    if (response.body) {
        const reader = response.body.getReader();
        let position = 0;
        const stream = new ReadableStream({
            start(controller) {
                const read = () => {
                    reader.read().then((result) => {
                        if (result.done) {
                            clearLine();
                            controller.close();
                            return;
                        }
                        position += result.value.length;
                        write('  ' + position + ' bytes\r');
                        controller.enqueue(result.value);
                        read();
                    }).catch(error => {
                        controller.error(error);
                    });
                };
                read();
            }
        });
        return new Response(stream, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers
        });
    }
    return response;
};

const download = async (url) => {
    writeLine('download ' + url);
    const response = await request(url);
    return response.arrayBuffer().then((buffer) => new Uint8Array(buffer));
};

const hash = async (url, algorithm) => {
    const data = await download(url);
    const hash = crypto.createHash(algorithm);
    hash.update(data);
    return hash.digest('hex');
};

const fork = async (organization, repository) => {
    const headers = {
        Authorization: 'Bearer ' + process.env.GITHUB_TOKEN
    };
    writeLine('github delete ' + repository);
    await request('https://api.github.com/repos/' + process.env.GITHUB_USER + '/homebrew-cask', {
        method: 'DELETE',
        headers: headers
    }, false);
    await sleep(4000);
    writeLine('github fork ' + repository);
    await request('https://api.github.com/repos/' + organization + '/' + repository + '/forks', {
        method: 'POST',
        headers: headers,
        body: ''
    });
    await sleep(4000);
    rm('dist', repository);
    writeLine('github clone ' + repository);
    exec('git clone --depth=2 https://x-access-token:' + process.env.GITHUB_TOKEN + '@github.com/' + process.env.GITHUB_USER + '/' + repository + '.git ' + 'dist/' + repository);
};

const pullrequest = async (organization, repository, body) => {
    writeLine('github push ' + repository);
    exec('git -C dist/' + repository + ' push');
    writeLine('github pullrequest homebrew-cask');
    const headers = {
        Authorization: 'Bearer ' + process.env.GITHUB_TOKEN
    };
    await request('https://api.github.com/repos/' + organization + '/' + repository + '/pulls', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(body)
    });
};

const install = () => {
    const node_modules = path.join(__dirname, 'node_modules');
    if (!fs.existsSync(node_modules)) {
        const options = { cwd: __dirname, stdio: [ 0,1,2 ] };
        child_process.execSync('npm install', options);
    }
};

const start = () => {
    install();
    exec('npx electron .');
}

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
            writeLine('build web');
            rm('dist', 'web');
            mkdir('dist', 'web');
            writeLine('cp source/dir dist/dir');
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
            writeLine('build electron');
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
            writeLine('build python');
            exec('python package.py build version');
            exec('python -m pip install --user build wheel --quiet');
            exec('python -m build --no-isolation --wheel --outdir dist/pypi dist/pypi');
            if (read('install')) {
                exec('python -m pip install --force-reinstall dist/pypi/*.whl');
            }
            break;
        }
        default: {
            writeLine('build');
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
            writeLine('publish web');
            build('web');
            rm('dist', 'gh-pages');
            const url = 'https://x-access-token:' + GITHUB_TOKEN + '@github.com/' + GITHUB_USER + '/netron.git';
            exec('git clone --depth=1 ' + url + ' --branch gh-pages ./dist/gh-pages 2>&1 > /dev/null');
            writeLine('cp dist/web dist/gh-pages');
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
            writeLine('publish electron');
            install();
            exec('npx electron-builder install-app-deps');
            exec('npx electron-builder --mac --universal --publish always');
            exec('npx electron-builder --win --x64 --arm64 --publish always');
            exec('npx electron-builder --linux appimage --x64 --publish always');
            exec('npx electron-builder --linux snap --x64 --publish always');
            break;
        }
        case 'python': {
            writeLine('publish python');
            build('python');
            exec('python -m pip install --user twine');
            exec('python -m twine upload --non-interactive --skip-existing --verbose dist/pypi/*.whl');
            break;
        }
        case 'cask': {
            writeLine('publish cask');
            await fork('Homebrew', 'homebrew-cask');
            const repository = 'https://github.com/' + configuration.repository;
            const url = repository + '/releases/download/v#{version}/' + configuration.productName + '-#{version}-mac.zip';
            const sha256 = await hash(url.replace(/#{version}/g, configuration.version), 'sha256');
            writeLine('update manifest');
            const file = path.join(mkdir('dist', 'homebrew-cask', 'Casks'), 'netron.rb');
            fs.writeFileSync(file, [
                'cask "' + configuration.name + '" do',
                '  version "' + configuration.version + '"',
                '  sha256 "' + sha256.toLowerCase() + '"',
                '',
                '  url "' + url + '"',
                '  name "' + configuration.productName + '"',
                '  desc "' + configuration.description + '"',
                '  homepage "' + repository + '"',
                '',
                '  auto_updates true',
                '',
                '  app "' + configuration.productName + '.app"',
                '',
                '  zap trash: [',
                '    "~/Library/Application Support/' + configuration.productName + '",',
                '    "~/Library/Preferences/' + configuration.build.appId + '.plist",',
                '    "~/Library/Saved Application State/' + configuration.build.appId + '.savedState",',
                '  ]',
                'end',
                ''
            ].join('\n'));
            writeLine('git push homebrew-cask');
            exec('git -C dist/homebrew-cask add --all');
            exec('git -C dist/homebrew-cask commit -m "Update ' + configuration.name + ' to ' + configuration.version + '"');
            await pullrequest('Homebrew', 'homebrew-cask', {
                title: 'Update ' + configuration.name + ' to ' + configuration.version,
                body: 'Update version and sha256',
                head: process.env.GITHUB_USER + ':master',
                base: 'master'
            });
            rm('dist', 'homebrew-cask');
            break;
        }
        case 'winget': {
            writeLine('publish winget');
            await fork('microsoft', 'winget-pkgs');
            const name = configuration.name;
            const version = configuration.version;
            const product = configuration.productName;
            const publisher = configuration.author.name;
            const identifier = publisher.replace(' ', '') + '.' + product;
            const copyright = 'Copyright (c) ' + publisher;
            const repository = 'https://github.com/' + configuration.repository;
            const url = repository + '/releases/download/v' + version + '/' + product + '-Setup-' + version + '.exe';
            const extensions = configuration.build.fileAssociations.map((entry) => '- ' + entry.ext).sort().join('\n');
            writeLine('download ' + url);
            const sha256 = await hash(url, 'sha256');
            const paths = [ 'dist', 'winget-pkgs', 'manifests', publisher[0].toLowerCase(), publisher.replace(' ', ''), product ];
            // rm(...paths);
            // exec('git -C dist/winget-pkgs add --all');
            // exec('git -C dist/winget-pkgs commit -m "Remove ' + configuration.name + '"');
            paths.push(version);
            mkdir(...paths);
            writeLine('update manifest');
            const manifestFile = path.join(__dirname, ...paths, identifier);
            fs.writeFileSync(manifestFile + '.yaml', [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.version.1.2.0.schema.json',
                'PackageIdentifier: ' + identifier,
                'PackageVersion: ' + version,
                'DefaultLocale: en-US',
                'ManifestType: version',
                'ManifestVersion: 1.2.0',
                ''
            ].join('\n'));
            fs.writeFileSync(manifestFile + '.installer.yaml', [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.installer.1.2.0.schema.json',
                'PackageIdentifier: ' + identifier,
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
                '  InstallerSha256: ' + sha256.toUpperCase(),
                '  InstallerLocale: en-US',
                '  InstallerSwitches:',
                '    Custom: /NORESTART',
                '  UpgradeBehavior: install',
                '- Architecture: arm64',
                '  Scope: user',
                '  InstallerType: nullsoft',
                '  InstallerUrl: ' + url,
                '  InstallerSha256: ' + sha256.toUpperCase(),
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
                'PackageIdentifier: ' + identifier,
                'PackageVersion: ' + version,
                'PackageName: ' + product,
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
            writeLine('git push winget-pkgs');
            exec('git -C dist/winget-pkgs add --all');
            exec('git -C dist/winget-pkgs commit -m "Update ' + configuration.name + ' to ' + configuration.version + '"');
            await pullrequest('microsoft', 'winget-pkgs', {
                title: 'Update ' + configuration.productName + ' to ' + configuration.version,
                body: '',
                head: process.env.GITHUB_USER + ':master',
                base: 'master'
            });
            rm('dist', 'winget-pkgs');
            break;
        }
        default: {
            writeLine('publish');
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
    writeLine('eslint');
    exec('npx eslint source/*.js test/*.js publish/*.js tools/*.js');
    writeLine('pylint');
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
            case 'start': start(); break;
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
        if (process.stdout.write) {
            process.stdout.write(err.message + os.EOL);
        }
        process.exit(1);
    }
};

load();
next();
