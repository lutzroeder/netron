
import * as fs from 'fs/promises';

const main = async () => {
    const [, , source, target] = process.argv;
    let content = await fs.readFile(source, 'utf-8');
    content = content.replace(/required float min = 1;/g, 'optional float min = 1;');
    content = content.replace(/required float max = 2;/g, 'optional float max = 2;');
    await fs.writeFile(target, content, 'utf-8');
};

await main();
