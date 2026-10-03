// Optional real-browser smoke test using Node >= 22 and installed Chrome/Edge.
// Start Dash on port 8097 first. No npm dependencies required.
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';

const executable = process.env.DASHBOARD_BROWSER || [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].find(existsSync);
if (!executable) throw new Error('Set DASHBOARD_BROWSER to your Chromium executable.');
const profile = await mkdtemp(join(tmpdir(), 'coursera-browser-'));
const browser = spawn(executable, ['--headless=new', '--remote-debugging-port=0',
  `--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check',
  '--disable-background-networking', '--disable-extensions', '--disable-gpu',
  '--disable-gpu-sandbox', '--use-angle=swiftshader', '--no-sandbox', '--single-process',
  '--remote-allow-origins=*',
  'about:blank'], { windowsHide: true });
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    const timer = setTimeout(() => reject(new Error(`Browser startup timeout: ${stderr.slice(-1000)}`)), 15000);
    browser.once('error', reject);
    browser.stderr.on('data', chunk => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) { clearTimeout(timer); resolve(match[1]); }
    });
  });
  socket = new WebSocket(endpoint);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  let sequence = 0;
  const pending = new Map();
  const errors = [];
  function receive(event) {
    const response = JSON.parse(event.data);
    if (response.id && pending.has(response.id)) {
      const { resolve, reject, timer } = pending.get(response.id);
      clearTimeout(timer); pending.delete(response.id);
      if (response.error) reject(new Error(JSON.stringify(response.error)));
      else resolve(response.result);
    }
    if (response.method === 'Runtime.exceptionThrown') errors.push(response.params.exceptionDetails);
  }
  socket.onmessage = receive;
  function command(method, params = {}, sessionId) {
    return new Promise((resolve, reject) => {
      const id = ++sequence;
      const timer = setTimeout(() => reject(new Error(`CDP timeout: ${method}`)), 20000);
      pending.set(id, { resolve, reject, timer });
      socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
  const { targetId } = await command('Target.createTarget', { url: 'about:blank' });
  const debugBase = endpoint.replace(/^ws:/, 'http:').replace(/\/devtools\/browser\/.*$/, '');
  const targets = await (await fetch(`${debugBase}/json/list`)).json();
  const pageEndpoint = targets.find(target => target.id === targetId)?.webSocketDebuggerUrl;
  if (!pageEndpoint) throw new Error('Cannot find page DevTools endpoint.');
  socket.close();
  socket = new WebSocket(pageEndpoint);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = receive;
  const send = (method, params) => command(method, params);
  await send('Runtime.enable');
  await send('Page.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1050, deviceScaleFactor: 1, mobile: false });
  async function evaluate(expression) {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  async function until(expression, timeout = 30000) {
    const started = Date.now();
    while (Date.now() - started < timeout) {
      if (await evaluate(expression)) return;
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    throw new Error(`Condition timeout: ${expression}`);
  }
  async function click(selector) {
    const point = await evaluate(`(() => {
      const e=document.querySelector(${JSON.stringify(selector)});
      e.scrollIntoView({block:'center'});
      if (e instanceof SVGGeometryElement) {
        const box=e.getBBox();
        for (let iy=1;iy<20;iy++) for(let ix=1;ix<20;ix++) {
          const p=new DOMPoint(box.x+box.width*ix/20,box.y+box.height*iy/20);
          if(e.isPointInFill(p)) { const q=p.matrixTransform(e.getScreenCTM()); return {x:q.x,y:q.y}; }
        }
      }
      const r=e.getBoundingClientRect(); return {x:r.x+r.width/2,y:r.y+r.height/2};
    })()`);
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', ...point });
    await new Promise(resolve => setTimeout(resolve, 100));
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', ...point, button: 'left', clickCount: 1 });
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', ...point, button: 'left', clickCount: 1 });
  }
  async function screenshot(name) {
    const { data } = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    await writeFile(join('Thach/outputs/dashboard-preview', name + '.png'), Buffer.from(data, 'base64'));
  }
  async function setViewport(width, height, deviceScaleFactor = 1) {
    await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor, mobile: false });
    await evaluate("window.dispatchEvent(new Event('resize'))");
    await new Promise(resolve => setTimeout(resolve, 900));
  }
  async function assertVisibleChartsFit(panelSelector, label, expectedColumns = null) {
    const metrics = await evaluate(`(() => {
      const panel = document.querySelector(${JSON.stringify(panelSelector)});
      const grid = panel?.querySelector('.chart-grid');
      const graphs = [...(panel?.querySelectorAll('.chart-card .js-plotly-plot') || [])]
        .filter(graph => graph.getBoundingClientRect().width > 0)
        .map(graph => {
          const card = graph.closest('.chart-card').getBoundingClientRect();
          const plot = graph.getBoundingClientRect();
          const svg = graph.querySelector('.svg-container')?.getBoundingClientRect();
          return {
            cardWidth: card.width,
            cardHeight: card.height,
            plotWidth: plot.width,
            plotHeight: plot.height,
            leftGap: plot.left - card.left,
            rightGap: card.right - plot.right,
            svgWidth: svg?.width || 0,
            svgHeight: svg?.height || 0,
          };
        });
      const columns = grid
        ? getComputedStyle(grid).gridTemplateColumns.split(' ').filter(Boolean).length
        : 0;
      return {
        graphs,
        columns,
        pageOverflow: document.documentElement.scrollWidth > window.innerWidth,
      };
    })()`);
    console.log(`Chart metrics [${label}]:`, JSON.stringify(metrics));
    assert(metrics.graphs.length > 0, `${label}: must have visible charts`);
    assert.equal(metrics.pageOverflow, false, `${label}: page must not overflow horizontally`);
    if (expectedColumns !== null) {
      assert.equal(metrics.columns, expectedColumns, `${label}: unexpected chart-grid column count`);
    }
    for (const chart of metrics.graphs) {
      assert(chart.leftGap >= -1, `${label}: plot crosses the card's left edge`);
      assert(chart.rightGap >= -1, `${label}: plot crosses the card's right edge`);
      assert(chart.plotWidth <= chart.cardWidth + 1, `${label}: plot is wider than its card`);
      assert(chart.svgWidth <= chart.plotWidth + 1, `${label}: Plotly SVG is wider than its wrapper`);
      assert(chart.plotHeight >= 390 && chart.plotHeight <= 410, `${label}: plot height must stay near 400px`);
      assert(chart.svgHeight <= 410, `${label}: Plotly SVG is too tall`);
      assert(chart.plotHeight <= chart.cardHeight + 1, `${label}: plot is taller than its card`);
    }
  }
  await mkdir('Thach/outputs/dashboard-preview', { recursive: true });
  await send('Page.navigate', { url: process.env.DASHBOARD_URL || 'http://127.0.0.1:8097' });
  await until(`document.querySelector('#kpi-row')?.innerText.includes('6,645') && document.querySelectorAll('#chart-level .point').length === 4`);
  await new Promise(resolve => setTimeout(resolve, 500));
  await screenshot('overview');
  await assertVisibleChartsFit('#overview-panel', 'desktop overview', 2);
  await click('#chart-level .point path');
  await until(`document.querySelector('#breadcrumb')?.innerText.includes('Beginner')`);
  await click('#page-tab .tab:nth-child(4)');
  await until(`document.querySelector('#data-panel')?.style.display === 'block' && document.querySelector('#table-caption')?.innerText.includes('3,591') && !!document.querySelector('#course-table td[data-dash-column="title"]')`);
  await click('#course-table td[data-dash-column="title"]');
  await until(`document.querySelector('#course-detail')?.innerText.includes('CHI TIẾT KHÓA HỌC')`);
  console.log('PASS: initial render, data tab, real bar click cross-filter, table drill-down');
  await click('#clear-drill');
  await until(`document.querySelector('#table-caption')?.innerText.includes('6,645')`);
  await click('#page-tab .tab:nth-child(2)');
  await until(`document.querySelector('#insight-panel')?.style.display === 'block' && document.querySelector('#chart-scatter .js-plotly-plot')?._fullData?.[0]?.x?.length > 0`);
  await assertVisibleChartsFit('#insight-panel', 'desktop insight', 2);
  await evaluate('window.scrollTo(0,0)');
  await screenshot('insight');
  // Emit the same Plotly event payload a picked WebGL point produces.
  await evaluate(`(() => { const g=document.querySelector('#chart-scatter .js-plotly-plot'); const t=g._fullData[0]; g.emit('plotly_click', {points:[{curveNumber:0,pointNumber:0,x:t.x[0],y:t.y[0],customdata:t.customdata[0]}]}); })()`);
  await click('#page-tab .tab:nth-child(4)');
  await until(`document.querySelector('#data-panel')?.style.display === 'block'`);
  await until(`document.querySelector('#course-detail')?.innerText.includes('CHI TIẾT KHÓA HỌC')`);
  await click('#page-tab .tab:nth-child(3)');
  await until(`document.querySelector('#prediction-panel')?.style.display === 'block' && document.querySelector('#model-estimate')?.innerText.includes('32,541')`);
  await evaluate('window.scrollTo(0,0)');
  await screenshot('prediction');
  console.log('PASS: insight tab, scatter detail, prediction tab and form result');
  await click('#page-tab .tab:nth-child(1)');
  await until(`document.querySelector('#overview-panel')?.style.display === 'block'`);
  const businessSelector = await evaluate(`(() => {
    const slice=[...document.querySelectorAll('#chart-tree .slice')]
      .find(node => node.textContent.toLowerCase().includes('business'));
    const path=slice?.querySelector('path.surface');
    if (!path) return null;
    path.id='smoke-business-slice';
    return '#smoke-business-slice';
  })()`);
  assert(businessSelector, 'Treemap must render a Business slice');
  await click(businessSelector);
  await until(`document.querySelector('#breadcrumb')?.innerText.includes('Chủ đề:')`);
  await click('#clear-drill');
  await until(`document.querySelector('#table-caption')?.innerText.includes('6,645')`);
  await until(`document.querySelector('#chart-map .js-plotly-plot')?._fullData?.[0]?.locations?.length > 0`);
  const mapGeometryReady = await evaluate(`!!document.querySelector('#chart-map .choroplethlayer path')`);
  if (mapGeometryReady) {
    await evaluate(`document.querySelector('#chart-map .js-plotly-plot').on('plotly_click', e => { window.__mapClick=e.points.map(p=>({location:p.location,pointNumber:p.pointNumber})); })`);
    await click('#chart-map .choroplethlayer path');
    await new Promise(resolve => setTimeout(resolve, 500));
    console.log('Map click:', await evaluate(`JSON.stringify({event:window.__mapClick,breadcrumb:document.querySelector('#breadcrumb').innerText})`));
    await screenshot('map-click');
    await until(`document.querySelector('#breadcrumb')?.innerText.includes('Trụ sở:')`);
  } else {
    console.log('SKIP: headless software renderer did not load Plotly world geometry; map data is still checked below.');
  }
  if (!await evaluate(`document.querySelector('#filter-panel details')?.open`)) {
    await click('#filter-panel summary');
    await until(`document.querySelector('#filter-panel details')?.open === true`);
  }
  await click('#reset-filters');
  await until(`document.querySelector('#table-caption')?.innerText.includes('6,645')`);
  console.log('PASS: real treemap click, map data and reset filters');
  const mapStatus = await evaluate(`(() => { const graph=document.querySelector('#chart-map .js-plotly-plot'); return {countries:document.querySelectorAll('#chart-map .choroplethlayer path').length, dataCountries:graph?._fullData?.[0]?.locations?.length || 0, geo:!!document.querySelector('#chart-map .geo')}; })()`);
  console.log('Map rendering:', JSON.stringify(mapStatus));
  assert.equal(mapStatus.dataCountries, 34, 'Choropleth must contain all mapped countries');
  await evaluate(`document.querySelector('#chart-map').scrollIntoView({block:'center'})`);
  await screenshot('map');

  await setViewport(1180, 900);
  for (const [tab, panel, label] of [
    [1, '#overview-panel', 'laptop overview'],
    [2, '#insight-panel', 'laptop insight'],
    [3, '#prediction-panel', 'laptop prediction'],
  ]) {
    await click(`#page-tab .tab:nth-child(${tab})`);
    await until(`document.querySelector(${JSON.stringify(panel)})?.style.display !== 'none'`);
    await assertVisibleChartsFit(panel, label, 1);
  }
  await screenshot('laptop');
  console.log('PASS: intermediate viewport uses one-column charts without overflow');

  // Gần với ảnh lỗi 1920×935 trên Windows scale 125%: 1536×748 CSS pixels.
  await setViewport(1536, 748, 1.25);
  await click('#page-tab .tab:nth-child(2)');
  await until(`document.querySelector('#insight-panel')?.style.display === 'block'`);
  await assertVisibleChartsFit('#insight-panel', '125% display scaling insight', 2);
  await screenshot('high-dpi-insight');
  console.log('PASS: 125% display scaling keeps insight charts at 400px');

  await setViewport(390, 844);
  await evaluate('window.scrollTo(0,0)');
  const overflow = await evaluate('document.documentElement.scrollWidth > window.innerWidth');
  assert.equal(overflow, false, 'Mobile must not overflow horizontally');
  const mobileTabsFit = await evaluate(`(() => {
    const sidebar = document.querySelector('#app-sidebar').getBoundingClientRect();
    return [...document.querySelectorAll('#page-tab .tab')].every(tab => {
      const box = tab.getBoundingClientRect();
      return box.left >= sidebar.left - 1 && box.right <= sidebar.right + 1;
    });
  })()`);
  assert.equal(mobileTabsFit, true, 'All four mobile tabs must stay inside the sidebar');
  await screenshot('mobile');
  assert.equal(errors.length, 0, JSON.stringify(errors));
  console.log('PASS: map, mobile width, no uncaught JavaScript errors');
} finally {
  socket?.close();
  browser.kill();
  browser.stderr.destroy();
  browser.stdout.destroy();
  browser.unref();
}
