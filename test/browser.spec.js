import * as fs from 'fs';
import * as path from 'path';
import * as playwright from '@playwright/test';
import * as url from 'url';

playwright.test.setTimeout(120000);

playwright.test('browser', async ({ page }) => {

    const self = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(self);
    const file = path.resolve(dir, '../third_party/test/onnx/candy.onnx');
    playwright.expect(fs.existsSync(file)).toBeTruthy();

    // Navigate to the application
    await page.goto('/');

    playwright.expect(page).toBeDefined();
    await page.waitForLoadState('domcontentloaded');

    // Wait for the welcome screen to be ready
    await page.waitForSelector('body.welcome', { timeout: 25000 });
    await page.waitForTimeout(1000);

    const consent = await page.locator('#message-button');
    if (await consent.isVisible({ timeout: 25000 })) {
        await consent.click();
    }

    // Set up file chooser promise before clicking
    const fileChooserPromise = page.waitForEvent('filechooser');
    const openButton = await page.locator('.open-file-button, button:has-text("Open Model")');
    await openButton.click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(file);

    // Wait for the graph to render
    await page.waitForSelector('#canvas', { state: 'attached', timeout: 10000 });
    await page.waitForSelector('body.default', { timeout: 10000 });

    // Open find sidebar
    const menuButton = await page.locator('#menu-button');
    await menuButton.click();
    await page.waitForTimeout(200);
    const findMenuItem = await page.locator('button:has-text("Find...")');
    await findMenuItem.click();
    await page.waitForTimeout(500);
    const search = await page.waitForSelector('#search', { state: 'visible', timeout: 5000 });
    playwright.expect(search).toBeDefined();

    // Find and activate tensor
    await search.fill('convolution1_W');
    await page.waitForSelector('.sidebar-find-content li', { state: 'attached' });
    const item = await page.waitForSelector('.sidebar-find-content li:has-text("convolution1_W")');
    await item.dblclick();

    // Expand the 'value' field
    const valueEntry = await page.waitForSelector('#sidebar-content .sidebar-item:has(.sidebar-item-name input[value="value"])');
    const valueButton = await valueEntry.waitForSelector('.sidebar-item-value-button');
    await valueButton.click();

    // Check first number from tensor value
    const pre = await valueEntry.waitForSelector('pre');
    const text = (await pre.textContent()) || '';
    const match = text.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/i);
    playwright.expect(match).not.toBeNull();
    const first = parseFloat(match[0]);
    playwright.expect(first).toBe(0.1353299617767334);
});

playwright.test('node neighborhood highlighting', async ({ page }) => {

    const self = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(self);
    const file = path.resolve(dir, 'neighborhood.dot');
    playwright.expect(fs.existsSync(file)).toBeTruthy();
    await page.emulateMedia({ colorScheme: 'light' });

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 25000 });

    const consent = page.locator('#message-button');
    if (await consent.isVisible()) {
        await consent.click();
    }

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.locator('.open-file-button, button:has-text("Open Model")').click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(file);
    await page.waitForSelector('body.default', { timeout: 10000 });

    await page.locator('#node-name-selected .node-item-type').click();

    const inputEdges = page.locator('.edge-path.input-highlight');
    const outputEdges = page.locator('.edge-path.output-highlight');
    await playwright.expect(page.locator('#node-name-selected')).toHaveClass(/\bselect\b/);
    await playwright.expect(page.locator('#node-name-input')).toHaveClass(/\binput-highlight\b/);
    await playwright.expect(page.locator('#node-name-output')).toHaveClass(/\boutput-highlight\b/);
    await playwright.expect(inputEdges).toHaveCount(1);
    await playwright.expect(outputEdges).toHaveCount(1);
    await playwright.expect(page.locator('#node-name-input > .node-border')).toHaveCSS('stroke', 'rgb(0, 138, 59)');
    await playwright.expect(page.locator('#node-name-selected > .node-border')).toHaveCSS('stroke', 'rgb(37, 99, 235)');
    await playwright.expect(page.locator('#node-name-output > .node-border')).toHaveCSS('stroke', 'rgb(211, 47, 47)');
    await playwright.expect(inputEdges).toHaveCSS('marker-end', /arrowhead-input-highlight/);
    await playwright.expect(outputEdges).toHaveCSS('marker-end', /arrowhead-output-highlight/);

    const canvas = page.locator('#canvas');
    const style = await canvas.getAttribute('style');
    await page.locator('#zoom-in-button').click();
    await playwright.expect.poll(async () => await canvas.getAttribute('style')).not.toBe(style);
    await playwright.expect(inputEdges).toHaveCount(1);
    await playwright.expect(outputEdges).toHaveCount(1);

    await page.locator('#node-name-output .node-item-type').click();
    await playwright.expect(page.locator('#node-name-input')).not.toHaveClass(/\binput-highlight\b/);
    await playwright.expect(page.locator('#node-name-selected')).toHaveClass(/\binput-highlight\b/);
    await playwright.expect(page.locator('#node-name-output > .node-border')).toHaveCSS('stroke', 'rgb(37, 99, 235)');
    await playwright.expect(outputEdges).toHaveCount(0);
});