
import * as child_process from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import * as url from 'url';

const args = process.argv.slice(2);

const read = (match) => {
    if (args.length > 0 || (!match || args[0] === match)) {
        return args.shift();
    }
    return null;
};

let configuration = null;

const dirname = (...args) => {
    const file = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(file);
    return path.join(dir, ...args);
};

const load = async () => {
    const file = dirname('package.json');
    const content = await fs.readFile(file, 'utf-8');
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

const access = async (path) => {
    try {
        await fs.access(path);
        return true;
    } catch {
        return false;
    }
};

const rm = async (...args) => {
    const dir = dirname(...args);
    const exists = await access(dir);
    if (exists) {
        const paths = path.join(...args);
        writeLine(`rm ${paths}`);
        const options = { recursive: true, force: true };
        await fs.rm(dir, options);
    }
};

const mkdir = async (...args) => {
    const dir = dirname(...args);
    const exists = await access(dir);
    if (!exists) {
        const paths = path.join(...args);
        writeLine(`mkdir ${paths}`);
        const options = { recursive: true };
        await fs.mkdir(dir, options);
    }
    return dir;
};

const copy = async (source, target, filter) => {
    let files = await fs.readdir(source);
    files = filter ? files.filter((file) => filter(file)) : files;
    const promises = files.map((file) => fs.copyFile(path.join(source, file), path.join(target, file)));
    await Promise.all(promises);
};

const unlink = async (dir, filter) => {
    let files = await fs.readdir(dir);
    files = filter ? files.filter((file) => filter(file)) : files;
    const promises = files.map((file) => fs.unlink(path.join(dir, file)));
    await Promise.all(promises);
};

const exec = async (command, encoding, cwd) => {
    cwd = cwd || dirname();
    if (encoding) {
        return child_process.execSync(command, { cwd, encoding });
    }
    child_process.execSync(command, { cwd, stdio: [0,1,2] });
    return '';
    /*
    return new Promise((resolve, reject) => {
        const child = child_process.exec(command, { cwd: dirname() }, (error, stdout, stderr) => {
            if (error) {
                stderr = '\n' + stderr ;
                if (error.message && error.message.endsWith(stderr)) {
                    error.message = error.message.slice(0, -stderr.length);
                }
                reject(error);
            } else {
                resolve(stdout);
            }
        });
        child.stdout.pipe(process.stdout);
        child.stderr.pipe(process.stderr);
    });
    */
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
                const read = async () => {
                    try {
                        const result = await reader.read();
                        if (result.done) {
                            clearLine();
                            controller.close();
                        } else {
                            position += result.value.length;
                            write(`  ${position} bytes\r`);
                            controller.enqueue(result.value);
                            read();
                        }
                    } catch (error) {
                        controller.error(error);
                    }
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
    writeLine(`download ${url}`);
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
        Authorization: `Bearer ${process.env.GITHUB_TOKEN}`
    };
    writeLine(`github delete ${repository}`);
    await request(`https://api.github.com/repos/${process.env.GITHUB_USER}/${repository}`, {
        method: 'DELETE',
        headers
    }, false);
    await sleep(4000);
    writeLine(`github fork ${repository}`);
    await request(`https://api.github.com/repos/${organization}/${repository}/forks`, {
        method: 'POST',
        headers,
        body: ''
    });
    await sleep(4000);
    await rm('dist', repository);
    writeLine(`github clone ${repository}`);
    await exec(`git clone --depth=2 https://x-access-token:${process.env.GITHUB_TOKEN}@github.com/${process.env.GITHUB_USER}/${repository}.git dist/${repository}`);
};

const pullrequest = async (organization, repository, body) => {
    writeLine(`github push ${repository}`);
    await exec(`git -C dist/${repository} push`);
    writeLine(`github pullrequest ${repository}`);
    const headers = {
        Authorization: `Bearer ${process.env.GITHUB_TOKEN}`
    };
    await request(`https://api.github.com/repos/${organization}/${repository}/pulls`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body)
    });
};

const clean = async () => {
    await rm('dist');
    await rm('node_modules');
    await rm('package-lock.json');
    await rm('yarn.lock');
};

const install = async () => {
    const node_modules = dirname('node_modules');
    let exists = await access(node_modules);
    if (exists) {
        const dependencies = { ...configuration.dependencies, ...configuration.devDependencies };
        const matches = await Promise.all(Object.entries(dependencies).map(async ([name, version]) => {
            const file = path.join('node_modules', name, 'package.json');
            const exists = await access(file);
            if (exists) {
                const content = await fs.readFile(file, 'utf8');
                const obj = JSON.parse(content);
                return obj.version === version;
            }
            return false;
        }));
        exists = matches.every((match) => match);
        if (!exists) {
            await clean();
        }
    }
    exists = await access(node_modules);
    if (!exists) {
        await exec('npm install');
    }
    try {
        await exec('python --version', 'utf-8');
        await exec('python -m pip install --upgrade --quiet setuptools pylint');
    } catch {
        // continue regardless of error
    }
};

const start = async () => {
    await install();
    await exec('npx electron .');
};

const purge = async () => {
    await clean();
    await rm('third_party', 'bin');
    await rm('third_party', 'env');
    await rm('third_party', 'source');
};

const build = async (target) => {
    switch (target || read()) {
        case 'web': {
            writeLine('build web');
            await rm('dist', 'web');
            await mkdir('dist', 'web');
            writeLine('cp source/dir dist/dir');
            const source_dir = dirname('source');
            const dist_dir = dirname('dist', 'web');
            const extensions = new Set(['html', 'css', 'js', 'json', 'ico', 'png']);
            await copy(source_dir, dist_dir, (file) => extensions.has(file.split('.').pop()));
            await rm('dist', 'web', 'app.js');
            await rm('dist', 'web', 'electron.js');
            const contentFile = dirname('dist', 'web', 'index.html');
            let content = await fs.readFile(contentFile, 'utf-8');
            content = content.replace(/(<meta\s*name="version"\s*content=")(.*)(">)/m, (match, p1, p2, p3) => {
                return p1 + configuration.version + p3;
            });
            content = content.replace(/(<meta\s*name="date"\s*content=")(.*)(">)/m, (match, p1, p2, p3) => {
                return p1 + configuration.date + p3;
            });
            await fs.writeFile(contentFile, content, 'utf-8');
            break;
        }
        case 'electron': {
            writeLine('build electron');
            await install();
            await exec('npx electron-builder install-app-deps');
            await exec('npx electron-builder --mac --universal --publish never -c.mac.identity=null');
            await exec('npx electron-builder --win --x64 --arm64 --publish never');
            await exec('npx electron-builder --linux appimage --x64 --publish never');
            await exec('npx electron-builder --linux snap --x64 --publish never');
            break;
        }
        case 'python': {
            writeLine('build python');
            await exec('python package.py build version');
            await exec('python -m pip install --user build wheel --quiet');
            await exec('python -m build --wheel --outdir dist/pypi dist/pypi');
            if (read('install')) {
                await exec('python -m pip install --force-reinstall dist/pypi/*.whl');
            }
            break;
        }
        default: {
            writeLine('build');
            await rm('dist');
            await install();
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
            await build('web');
            await rm('dist', 'gh-pages');
            const url = `https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/netron.git`;
            await exec(`git clone --depth=1 ${url} --branch gh-pages ./dist/gh-pages 2>&1 > /dev/null`);
            writeLine('cp dist/web dist/gh-pages');
            const source_dir = dirname('dist', 'web');
            const target_dir = dirname('dist', 'gh-pages');
            await unlink(target_dir, (file) => file !== '.git');
            await copy(source_dir, target_dir);
            await exec('git -C dist/gh-pages add --all');
            await exec('git -C dist/gh-pages commit --amend --no-edit');
            await exec('git -C dist/gh-pages push --force origin gh-pages');
            break;
        }
        case 'electron': {
            writeLine('publish electron');
            await install();
            await exec('npx electron-builder install-app-deps');
            await exec('npx electron-builder --mac --universal --publish always');
            await exec('npx electron-builder --win --x64 --arm64 --publish always');
            await exec('npx electron-builder --linux appimage --x64 --publish always');
            await exec('npx electron-builder --linux snap --x64 --publish always');
            break;
        }
        case 'python': {
            writeLine('publish python');
            await build('python');
            await exec('python -m pip install --user twine');
            await exec('python -m twine upload --non-interactive --skip-existing --verbose dist/pypi/*.whl');
            break;
        }
        case 'cask': {
            writeLine('publish cask');
            await fork('Homebrew', 'homebrew-cask');
            const repository = `https://github.com/${configuration.repository}`;
            const url = `${repository}/releases/download/v#{version}/${configuration.productName}-#{version}-mac.zip`;
            const sha256 = await hash(url.replace(/#{version}/g, configuration.version), 'sha256');
            writeLine('update manifest');
            const dir = await mkdir('dist', 'homebrew-cask', 'Casks', 'n');
            const file = path.join(dir, 'netron.rb');
            await fs.writeFile(file, [
                `cask "${configuration.name}" do`,
                `  version "${configuration.version}"`,
                `  sha256 "${sha256.toLowerCase()}"`,
                '',
                `  url "${url}"`,
                `  name "${configuration.productName}"`,
                `  desc "${configuration.description.replace('Visualizer', 'Visualiser')}"`,
                `  homepage "${repository}"`,
                '',
                '  auto_updates true',
                '',
                `  app "${configuration.productName}.app"`,
                '',
                '  zap trash: [',
                `    "~/Library/Application Support/${configuration.productName}",`,
                `    "~/Library/Preferences/${configuration.build.appId}.plist",`,
                `    "~/Library/Saved Application State/${configuration.build.appId}.savedState",`,
                '  ]',
                'end',
                ''
            ].join('\n'));
            writeLine('git push homebrew-cask');
            await exec('git -C dist/homebrew-cask add --all');
            await exec(`git -C dist/homebrew-cask commit -m "${configuration.name} ${configuration.version}"`);
            await pullrequest('Homebrew', 'homebrew-cask', {
                title: `${configuration.name} ${configuration.version}`,
                body: 'Update version and sha256',
                head: `${process.env.GITHUB_USER}:master`,
                base: 'master'
            });
            await rm('dist', 'homebrew-cask');
            break;
        }
        case 'winget': {
            writeLine('publish winget');
            await fork('microsoft', 'winget-pkgs');
            const name = configuration.name;
            const version = configuration.version;
            const product = configuration.productName;
            const publisher = configuration.author.name;
            const identifier = `${publisher.replace(' ', '')}.${product}`;
            const copyright = `Copyright (c) ${publisher}`;
            const repository = `https://github.com/${configuration.repository}`;
            const url = `${repository}/releases/download/v${version}/${product}-Setup-${version}.exe`;
            const content = await fs.readFile(configuration.build.extends, 'utf-8');
            const builder = JSON.parse(content);
            const extensions = builder.fileAssociations.map((entry) => `- ${entry.ext}`).sort().join('\n');
            const sha256 = await hash(url, 'sha256');
            const paths = ['dist', 'winget-pkgs', 'manifests', publisher[0].toLowerCase(), publisher.replace(' ', ''), product, version];
            await mkdir(...paths);
            writeLine('update manifest');
            const manifestFile = dirname(...paths, identifier);
            await fs.writeFile(`${manifestFile}.yaml`, [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.version.1.6.0.schema.json',
                `PackageIdentifier: ${identifier}`,
                `PackageVersion: ${version}`,
                'DefaultLocale: en-US',
                'ManifestType: version',
                'ManifestVersion: 1.6.0',
                ''
            ].join('\n'));
            await fs.writeFile(`${manifestFile}.installer.yaml`, [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.installer.1.6.0.schema.json',
                `PackageIdentifier: ${identifier}`,
                `PackageVersion: ${version}`,
                'Platform:',
                '- Windows.Desktop',
                'InstallModes:',
                '- silent',
                '- silentWithProgress',
                'Installers:',
                '- Architecture: x86',
                '  Scope: user',
                '  InstallerType: nullsoft',
                `  InstallerUrl: ${url}`,
                `  InstallerSha256: ${sha256.toUpperCase()}`,
                '  InstallerLocale: en-US',
                '  InstallerSwitches:',
                '    Custom: /NORESTART',
                '  UpgradeBehavior: install',
                '- Architecture: arm64',
                '  Scope: user',
                '  InstallerType: nullsoft',
                `  InstallerUrl: ${url}`,
                `  InstallerSha256: ${sha256.toUpperCase()}`,
                '  InstallerLocale: en-US',
                '  InstallerSwitches:',
                '    Custom: /NORESTART',
                '  UpgradeBehavior: install',
                'FileExtensions:',
                extensions,
                'ManifestType: installer',
                'ManifestVersion: 1.6.0',
                ''
            ].join('\n'));
            await fs.writeFile(`${manifestFile}.locale.en-US.yaml`, [
                '# yaml-language-server: $schema=https://aka.ms/winget-manifest.defaultLocale.1.6.0.schema.json',
                `PackageIdentifier: ${identifier}`,
                `PackageVersion: ${version}`,
                `PackageName: ${product}`,
                'PackageLocale: en-US',
                `PackageUrl: ${repository}`,
                `Publisher: ${publisher}`,
                `PublisherUrl: ${repository}`,
                `PublisherSupportUrl: ${repository}/issues`,
                `Author: ${publisher}`,
                `License: ${configuration.license}`,
                `Copyright: ${copyright}`,
                `CopyrightUrl: ${repository}/blob/main/LICENSE`,
                `ShortDescription: ${configuration.description}`,
                `Description: ${configuration.description}`,
                `Moniker: ${name}`,
                'Tags:',
                '- machine-learning',
                '- deep-learning',
                '- neural-network',
                'ManifestType: defaultLocale',
                'ManifestVersion: 1.6.0',
                ''
            ].join('\n'));
            writeLine('git push winget-pkgs');
            await exec('git -C dist/winget-pkgs add --all');
            await exec(`git -C dist/winget-pkgs commit -m "Update ${configuration.name} to ${configuration.version}"`);
            await pullrequest('microsoft', 'winget-pkgs', {
                title: `Update ${configuration.productName} to ${configuration.version}`,
                body: '',
                head: `${process.env.GITHUB_USER}:master`,
                base: 'master'
            });
            await rm('dist', 'winget-pkgs');
            break;
        }
        default: {
            writeLine('publish');
            await rm('dist');
            await install();
            await publish('web');
            await publish('electron');
            await publish('python');
            await publish('cask');
            await publish('winget');
            break;
        }
    }
};

const lint = async () => {
    await install();
    writeLine('eslint');
    await exec('npx eslint --config publish/eslint.config.js *.*js source/*.*js test/*.*js publish/*.*js tools/*.js');
    writeLine('pylint');
    await exec('python -m pylint -sn --recursive=y source test publish tools *.py');
};

const validate = async() => {
    writeLine('test');
    await exec('node test/models.js tag:validation');
    await lint();
};

const update = async () => {
    const dependencies = { ...configuration.dependencies, ...configuration.devDependencies };
    for (const name of Object.keys(dependencies)) {
        writeLine(name);
        /* eslint-disable no-await-in-loop */
        await exec(`npm install --quiet --no-progress --silent --save-exact ${name}@latest`);
        /* eslint-enable no-await-in-loop */
    }
    await install();
    const targets = process.argv.length > 3 ? process.argv.slice(3) : [
        'armnn',
        'bigdl',
        'caffe', 'circle', 'cntk', 'coreml',
        'dlc', 'dnn',
        'gguf',
        'kann', 'keras',
        'mnn', 'mslite', 'megengine',
        'nnabla',
        'onnx', 'om',
        'paddle', 'pytorch',
        'rknn',
        'sentencepiece', 'sklearn',
        'tf',
        'uff',
        'xmodel'
    ];
    for (const target of targets) {
        /* eslint-disable no-await-in-loop */
        await exec(`tools/${target} sync install schema metadata`);
        /* eslint-enable no-await-in-loop */
    }
};

const pull = async () => {
    await exec('git fetch --prune origin "refs/tags/*:refs/tags/*"');
    const before = await exec('git rev-parse HEAD', 'utf-8');
    try {
        await exec('git pull --prune --rebase');
    } catch (error) {
        writeLine(error.message);
    }
    const after = await exec('git rev-parse HEAD', 'utf-8');
    if (before.trim() !== after.trim()) {
        const output = await exec(`git diff --name-only ${before.trim()} ${after.trim()}`, 'utf-8');
        const files = new Set(output.split('\n'));
        if (files.has('package.json')) {
            await clean();
            await install();
        }
    }
};

const coverage = async () => {
    await rm('dist', 'nyc');
    await mkdir('dist', 'nyc');
    await exec('cp package.json dist/nyc');
    await exec('cp -R source dist/nyc');
    await exec('nyc instrument --compact false source dist/nyc/source');
    await exec('nyc --instrument npx electron ./dist/nyc');
};

const forge = async() => {
    const command = read();
    switch (command) {
        case 'install': {
            const packages = [
                '@electron-forge/cli',
                '@electron-forge/core',
                '@electron-forge/maker-snap',
                '@electron-forge/maker-dmg',
                '@electron-forge/maker-zip'
            ];
            await exec(`npm install ${packages.join(' ')} --no-save`);
            break;
        }
        case 'update': {
            const cwd = path.join(dirname(), '..', 'forge');
            const node_modules = path.join(cwd, 'node_modules');
            const links = path.join(cwd, '.links');
            const exists = await access(node_modules);
            if (!exists) {
                await exec('yarn', null, cwd);
            }
            await exec('yarn build', null, cwd);
            await exec('yarn link:prepare', null, cwd);
            await exec(`yarn link @electron-forge/core --link-folder=${links}`);
            break;
        }
        case 'build': {
            await exec('npx electron-forge make');
            break;
        }
        default: {
            throw new Error(`Unsupported forge command ${command}.`);
        }
    }
};

const analyze = async () => {
    const exists = await access('third_party/tools/codeql');
    if (!exists) {
        await exec('git clone --depth=1 https://github.com/github/codeql.git third_party/tools/codeql');
    }
    await rm('dist', 'codeql');
    await mkdir('dist', 'codeql', 'netron');
    await exec('cp -r publish source test tools dist/codeql/netron/');
    await exec('codeql database create dist/codeql/database --source-root dist/codeql/netron --language=javascript --threads=3');
    await exec('codeql database analyze dist/codeql/database ./third_party/tools/codeql/javascript/ql/src/codeql-suites/javascript-security-and-quality.qls --format=csv --output=dist/codeql/results.csv --threads=3');
    await exec('cat dist/codeql/results.csv');
};

const version = async () => {
    await pull();
    const file = dirname('package.json');
    let content = await fs.readFile(file, 'utf-8');
    content = content.replace(/(\s*"version":\s")(\d\.\d\.\d)(",)/m, (match, p1, p2, p3) => {
        const version = Array.from((parseInt(p2.split('.').join(''), 10) + 1).toString()).join('.');
        return p1 + version + p3;
    });
    content = content.replace(/(\s*"date":\s")(.*)(",)/m, (match, p1, p2, p3) => {
        const date = new Date().toISOString().split('.').shift().split('T').join(' ');
        return p1 + date + p3;
    });
    await fs.writeFile(file, content, 'utf-8');
    await load();
    await exec('git add package.json');
    await exec(`git commit -m "Update to ${configuration.version}"`);
    await exec(`git tag v${configuration.version}`);
    await exec('git push');
    await exec('git push --tags');
};

const main = async () => {
    await load();
    try {
        const task = read();
        switch (task) {
            case 'start': await start(); break;
            case 'clean': await clean(); break;
            case 'purge': await purge(); break;
            case 'install': await install(); break;
            case 'build': await build(); break;
            case 'publish': await publish(); break;
            case 'version': await version(); break;
            case 'lint': await lint(); break;
            case 'validate': await validate(); break;
            case 'update': await update(); break;
            case 'pull': await pull(); break;
            case 'analyze': await analyze(); break;
            case 'coverage': await coverage(); break;
            case 'forge': await forge(); break;
            default: throw new Error(`Unsupported task '${task}'.`);
        }
    } catch (error) {
        if (process.stdout.write) {
            process.stdout.write(error.message + os.EOL);
        }
        process.exit(1);
    }
};

await main();
