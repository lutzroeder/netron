
const fs = require('fs');

const file = process.argv[2];
const manifest = fs.readFileSync(file, 'utf-8');

const lines = manifest.split('\n');
const regexp = new RegExp(/(\s*"version":\s")(\d\.\d\.\d)(",)/);
for (let i = 0; i < lines.length; i++) {
    const line = lines[i].replace(regexp, (match, p1, p2, p3) => {
        const version = Array.from((parseInt(p2.split('.').join(''), 10) + 1).toString()).join('.');
        return p1 + version + p3;
    });
    if (line !== lines[i]) {
        lines[i] = line;
        break;
    }
}

fs.writeFileSync(file, lines.join('\n'), 'utf-8');
