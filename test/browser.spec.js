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

playwright.test('error handling - corrupted file shows error and returns to welcome', async ({ page }) => {
    const self = url.fileURLToPath(import.meta.url);
    const dir = path.dirname(self);
    const corruptedFile = path.resolve(dir, 'corrupted.onnx');

    playwright.expect(fs.existsSync(corruptedFile)).toBeTruthy();

    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    await page.waitForSelector('body.welcome', { timeout: 25000 });
    await page.waitForTimeout(1000);

    const consent = await page.locator('#message-button');
    if (await consent.isVisible({ timeout: 25000 })) {
        await consent.click();
    }

    const openButton = await page.locator('.open-file-button, button:has-text("Open Model")');
    playwright.expect(await openButton.isVisible()).toBeTruthy();

    const fileChooserPromise = page.waitForEvent('filechooser');
    await openButton.click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(corruptedFile);

    await page.waitForSelector('body.welcome.spinner', { timeout: 5000 });

    await page.waitForSelector('body.notification, body.alert', { timeout: 15000 });

    const messageText = await page.locator('#message-text');
    const errorContent = await messageText.textContent();
    playwright.expect(errorContent).not.toBe('');
    playwright.expect(errorContent.length).toBeGreaterThan(0);

    const messageButton = await page.locator('#message-button');
    playwright.expect(await messageButton.isVisible()).toBeTruthy();
    await messageButton.click();

    await page.waitForSelector('body.welcome', { timeout: 5000 });

    const bodyClass = await page.getAttribute('body', 'class');
    playwright.expect(bodyClass).toContain('welcome');
    playwright.expect(bodyClass).not.toContain('spinner');
    playwright.expect(bodyClass).not.toContain('notification');
    playwright.expect(bodyClass).not.toContain('alert');

    const openButtonAfterError = await page.locator('.open-file-button, button:has-text("Open Model")');
    playwright.expect(await openButtonAfterError.isVisible()).toBeTruthy();
    playwright.expect(await openButtonAfterError.isEnabled()).toBeTruthy();
});