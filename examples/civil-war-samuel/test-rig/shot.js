const { chromium } = require('playwright');
const [cookie, zoom, out, chapter] = process.argv.slice(2);
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1920, height: 1080 } });
  await ctx.addCookies([{ name: 'bs_session', value: cookie, url: 'http://127.0.0.1:8077' }]);
  await ctx.addInitScript((c) => { try { localStorage.setItem('bs_chapter_samuel', c); } catch (e) {} }, String(chapter || 1));
  const p = await ctx.newPage();
  await p.goto('http://127.0.0.1:8077/', { waitUntil: 'load' });
  await p.waitForTimeout(2500);
  if (zoom && zoom !== '-') {
    await p.evaluate(async (z) => { await fetch('/api/chat', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: `set the book to ${z} percent`, book: 'samuel' }) }); }, zoom);
    await p.reload({ waitUntil: 'load' }); await p.waitForTimeout(2500);
  }
  await p.screenshot({ path: out });
  await b.close();
})();
