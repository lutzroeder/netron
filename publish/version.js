
const fs = require('fs');

let content = fs.readFileSync(process.argv[2], 'utf-8');

content = content.replace(/(\s*"version":\s")(\d\.\d\.\d)(",)/m, (match, p1, p2, p3) => {
    const version = Array.from((parseInt(p2.split('.').join(''), 10) + 1).toString()).join('.');
    return p1 + version + p3;
});
content = content.replace(/(\s*"date":\s")(.*)(",)/m, (match, p1, p2, p3) => {
    const date = new Date().toISOString().split('.').shift().split('T').join(' ');
    return p1 + date + p3;
});

fs.writeFileSync(process.argv[2], content, 'utf-8');
