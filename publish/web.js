
const fs = require('fs');

const manifest = JSON.parse(fs.readFileSync(process.argv[2], 'utf-8'));

const file = process.argv[3];
let content = fs.readFileSync(file, 'utf-8');

content = content.replace(/(<meta\s*name="version"\s*content=")(.*)(">)/m, (match, p1, p2, p3) => {
    return p1 + manifest.version + p3;
});

content = content.replace(/(<meta\s*name="date"\s*content=")(.*)(">)/m, (match, p1, p2, p3) => {
    return p1 + manifest.date + p3;
});

fs.writeFileSync(file, content, 'utf-8');
