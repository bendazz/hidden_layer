// Feedforward Network Visualizer (3 inputs -> 4 hidden -> 1 output)
// Plain JS, no frameworks. Adjust data and weights below.

// ----------------------
// 1) Data (X0, X1, X2) and labels y
// ----------------------
// We'll generate a tiny binary dataset: two clusters in X1-X2 with bias X0=1
const data = [
  { x0: 1, x1: -1.2, x2: -0.8, y: 0 },
  { x0: 1, x1: -1.0, x2: -0.2, y: 0 },
  { x0: 1, x1: -0.5, x2: -1.2, y: 0 },
  { x0: 1, x1: -0.9, x2: -0.6, y: 0 },
  { x0: 1, x1: 1.3, x2: 0.9, y: 1 },
  { x0: 1, x1: 1.0, x2: 0.2, y: 1 },
  { x0: 1, x1: 0.5, x2: 1.1, y: 1 },
  { x0: 1, x1: 0.8, x2: 0.7, y: 1 },
];

// ----------------------
// 2) Network architecture and initial weights
// ----------------------
// Layer sizes: input=3 (x0, x1, x2), hidden=4, output=1
// We include biases as part of weight matrices by augmenting inputs with x0=1;
// to keep with user's request: X0 is the bias input feature (fixed 1s).

// Weights: input(3) -> hidden(4)
// w1[h][i] = weight from input i to hidden h; order inputs: [x0, x1, x2]
const w1 = [
  // h0
  [0.2, 0.8, -0.3],
  // h1
  [-0.1, -0.6, 0.7],
  // h2
  [0.05, 0.4, 0.4],
  // h3
  [0.3, -0.2, -0.5],
];

// Weights: hidden(4) -> output(1)
// w2[o][h] = weight from hidden h to output o (only o=0)
const w2 = [
  [0.7, -0.4, 0.5, -0.2],
];

// Activation functions
const relu = (z) => Math.max(0, z);
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// Forward pass for one sample
function forward(x0, x1, x2) {
  const x = [x0, x1, x2];
  const h = new Array(4);
  for (let j = 0; j < 4; j++) {
    let z = 0;
    for (let i = 0; i < 3; i++) z += w1[j][i] * x[i];
    h[j] = relu(z);
  }
  let zOut = 0;
  for (let j = 0; j < 4; j++) zOut += w2[0][j] * h[j];
  const yhat = sigmoid(zOut);
  return { h, yhat };
}

// Log loss across dataset
function logLoss(dataset) {
  const eps = 1e-9;
  let sum = 0;
  for (const d of dataset) {
    const { yhat } = forward(d.x0, d.x1, d.x2);
    const y = d.y;
    sum += -(y * Math.log(yhat + eps) + (1 - y) * Math.log(1 - yhat + eps));
  }
  return sum / dataset.length;
}

// ----------------------
// 3) Render data table
// ----------------------
function renderTable() {
  const tbody = document.querySelector('#data-table tbody');
  tbody.innerHTML = '';
  for (const d of data) {
    const tr = document.createElement('tr');
    for (const key of ['x0', 'x1', 'x2']) {
      const td = document.createElement('td');
      td.textContent = formatNum(d[key]);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function formatNum(v) {
  return Math.abs(v) < 1e-6 ? '0.000' : v.toFixed(3);
}

// ----------------------
// 4) Render network SVG
// ----------------------
function renderNetwork() {
  const svg = document.getElementById('network-svg');
  // Clear
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  // Layout positions
  const width = svg.viewBox.baseVal.width || 640;
  const height = svg.viewBox.baseVal.height || 420;
  const colX = [80, 300, 520]; // inputs, hidden, output

  const inY = [100, 200, 300]; // x0, x1, x2
  const hidY = [60, 160, 260, 360];
  const outY = [210];
  const r = 18;

  // Draw edges input->hidden
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 3; i++) {
      const w = w1[j][i];
      drawEdge(svg, colX[0] + r, inY[i], colX[1] - r, hidY[j], w);
      const mx = (colX[0] + colX[1]) / 2;
      const my = (inY[i] + hidY[j]) / 2 - 6;
  // weight labels removed for declutter
    }
  }

  // Draw edges hidden->output
  for (let j = 0; j < 4; j++) {
    const w = w2[0][j];
    drawEdge(svg, colX[1] + r, hidY[j], colX[2] - r, outY[0], w);
    const mx = (colX[1] + colX[2]) / 2;
    const my = (hidY[j] + outY[0]) / 2 - 6;
  // weight labels removed for declutter
  }

  // Draw nodes
  // Input nodes
  const inputs = ['X0', 'X1', 'X2'];
  for (let i = 0; i < 3; i++) drawNode(svg, colX[0], inY[i], r, inputs[i]);

  // Hidden nodes
  for (let j = 0; j < 4; j++) drawNode(svg, colX[1], hidY[j], r, `H${j}`);

  // Output node
  drawNode(svg, colX[2], outY[0], r, 'Ŷ');

  // Column labels (above)
  drawLabel(svg, colX[0], 24, 'Inputs');
  drawLabel(svg, colX[1], 24, 'Hidden');
  drawLabel(svg, colX[2], 24, 'Output');
}

function drawEdge(svg, x1, y1, x2, y2, w) {
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
  line.setAttribute('class', `edge ${w >= 0 ? 'positive' : 'negative'}`);
  line.setAttribute('stroke-width', String(1.2 + Math.min(3, Math.abs(w))));
  svg.appendChild(line);
}

function drawWeightLabel(svg, x, y, w) {
  const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  text.setAttribute('x', x);
  text.setAttribute('y', y);
  text.setAttribute('class', 'weight-label');
  text.textContent = formatNum(w);
  svg.appendChild(text);
}

function drawNode(svg, cx, cy, r, label) {
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  circle.setAttribute('cx', cx);
  circle.setAttribute('cy', cy);
  circle.setAttribute('r', r);
  circle.setAttribute('class', 'node');
  g.appendChild(circle);
  const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  text.setAttribute('x', cx);
  text.setAttribute('y', cy + 0.5);
  text.setAttribute('class', 'node-label');
  text.textContent = label;
  g.appendChild(text);
  svg.appendChild(g);
  return g;
}

function drawLabel(svg, x, y, label) {
  const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  text.setAttribute('x', x);
  text.setAttribute('y', y);
  text.setAttribute('class', 'neuron-label');
  text.textContent = label;
  svg.appendChild(text);
}

// ----------------------
// 5) Compute and render loss
// ----------------------
function renderLoss() {
  const L = logLoss(data);
  document.getElementById('loss-value').textContent = L.toFixed(4);

  // Show per-sample ŷ for transparency
  const lines = data.map((d, idx) => {
    const { yhat } = forward(d.x0, d.x1, d.x2);
    return `#${idx + 1}: y=${d.y}, ŷ=${yhat.toFixed(3)}`;
  });
  document.getElementById('loss-details').textContent = lines.join('\n');
}

// ----------------------
// 6) Boot
// ----------------------
function init() {
  renderTable();
  renderNetwork();
  renderWeights();
  renderLoss();
  renderFinalAnswerVectors();
}

window.addEventListener('DOMContentLoaded', init);

// ----------------------
// 8) Render weights table
// ----------------------
function renderWeights() {
  const w1Body = document.getElementById('w1-body');
  const w2Body = document.getElementById('w2-body');
  if (!w1Body || !w2Body) return;
  w1Body.innerHTML = '';
  w2Body.innerHTML = '';
  // w1 rows per hidden neuron
  for (let j = 0; j < w1.length; j++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.textContent = `H${j}`;
    th.style.textAlign = 'left';
    tr.appendChild(th);
    for (let i = 0; i < w1[j].length; i++) {
      const td = document.createElement('td');
      td.textContent = formatNum(w1[j][i]);
      tr.appendChild(td);
    }
    w1Body.appendChild(tr);
  }
  // w2 single row (output)
  const tr2 = document.createElement('tr');
  const th2 = document.createElement('th');
  th2.textContent = 'Ŷ';
  th2.style.textAlign = 'left';
  tr2.appendChild(th2);
  for (let j = 0; j < w2[0].length; j++) {
    const td = document.createElement('td');
    td.textContent = formatNum(w2[0][j]);
    tr2.appendChild(td);
  }
  w2Body.appendChild(tr2);
}

// ----------------------
// 9) Populate numeric vectors for Q2 and Q3 final answers
// ----------------------
function renderFinalAnswerVectors() {
  const q2El = document.getElementById('q2-final-vector');
  const q3El = document.getElementById('q3-final-vector');
  const q4El = document.getElementById('q4-final-vector');
  const q5El = document.getElementById('q5-final-vector');
  const q6El = document.getElementById('q6-final-vector');
  const q7El = document.getElementById('q7-updated-weight');
  const q8El = document.getElementById('q8-final-vector');
  const q9El = document.getElementById('q9-final-vector');
  const q10W2El = document.getElementById('q10-grad-w2');
  const q10W1El = document.getElementById('q10-grad-w1');
  const q11UpdW2El = document.getElementById('q11-upd-w2');
  const q11UpdW1El = document.getElementById('q11-upd-w1');
  if (!q2El && !q3El && !q4El && !q5El && !q6El && !q7El && !q8El && !q9El && !q10W2El && !q10W1El && !q11UpdW2El && !q11UpdW1El) return;
  const eps = 1e-9;
  const yVec = [];
  const yhatVec = [];
  const h0Vec = [];
  const x0Vec = [];
  const hMat = []; // store all hidden activations per sample
  for (const d of data) {
    const { h, yhat } = forward(d.x0, d.x1, d.x2);
    yVec.push(d.y);
    yhatVec.push(yhat);
    h0Vec.push(h[0]);
    x0Vec.push(d.x0);
    hMat.push(h);
  }
  // Q2: dL/dŷ vector
  if (q2El) {
    const dLdYhatVec = yhatVec.map((yhat, idx) => {
      const y = yVec[idx];
      const clip = Math.min(1 - eps, Math.max(eps, yhat));
      return (clip - y) / (clip * (1 - clip));
    });
    q2El.innerHTML = `[${dLdYhatVec.map(v=>v.toFixed(4)).join(', ')}]`;
  }
  // Q3: ∂ŷ/∂H0 vector = ŷ(1-ŷ) * w20
  if (q3El) {
    const w20 = w2[0][0];
    const dy_dH0 = yhatVec.map(yhat => yhat * (1 - yhat) * w20);
    q3El.innerHTML = `[${dy_dH0.map(v=>v.toFixed(4)).join(', ')}]`;
  }
  // Q4: ∂L/∂H0 = (ŷ - y) w20
  if (q4El) {
    const w20 = w2[0][0];
    const dL_dH0 = yhatVec.map((yhat, idx) => (yhat - yVec[idx]) * w20);
    q4El.innerHTML = `[${dL_dH0.map(v=>v.toFixed(4)).join(', ')}]`;
  }
  if (q5El) {
    // ∂ŷ/∂w20 = ŷ(1-ŷ) * H0
    const dy_dw20 = yhatVec.map((yhat, idx) => yhat * (1 - yhat) * h0Vec[idx]);
    q5El.innerHTML = `[${dy_dw20.map(v=>v.toFixed(4)).join(', ')}]`;
  }
  if (q6El) {
    // ∂L/∂w20 = ∑ over samples of (∂L/∂ŷ) * (∂ŷ/∂w20)
    // We show per-sample contributions: (ŷ - y) * H0
    // because (∂L/∂ŷ) = (ŷ - y)/(ŷ(1-ŷ)) and ∂ŷ/∂w20 = ŷ(1-ŷ)H0, factors cancel.
    const dL_dw20_vec = yhatVec.map((yhat, idx) => (yhat - yVec[idx]) * h0Vec[idx]);
    q6El.innerHTML = `[${dL_dw20_vec.map(v=>v.toFixed(4)).join(', ')}]`;
    // If Q7 exists, compute the averaged gradient and the updated weight
    if (q7El) {
      const N = dL_dw20_vec.length || 1;
      const avgGrad = dL_dw20_vec.reduce((a,b)=>a+b, 0) / N;
      const lr = 0.23;
      const w20_old = w2[0][0];
      const w20_new = w20_old - lr * avgGrad;
      q7El.innerHTML = `w₂₀(old) = ${w20_old.toFixed(4)} → w₂₀(new) = ${w20_new.toFixed(4)} (avg grad = ${avgGrad.toFixed(6)})`;
    }
  }
  // Q7: ∂H0/∂w10 = ReLU'(z0) * x0 = 1{z0>0} * x0
  if (q8El) {
    const w10 = w1[0][0];
    const w11 = w1[0][1];
    const w12 = w1[0][2];
    const dH0_dw10 = data.map(d => {
      const z0 = w10 * d.x0 + w11 * d.x1 + w12 * d.x2;
      const reluPrime = z0 > 0 ? 1 : 0;
      return reluPrime * d.x0;
    });
    q8El.innerHTML = `[${dH0_dw10.map(v=>v.toFixed(4)).join(', ')}]`;
  }
  // Q8: ∂L/∂w10 per-sample = (ŷ - y) * w20 * ReLU'(z0) * x0
  if (q9El) {
    const w20 = w2[0][0];
    const w10 = w1[0][0];
    const w11 = w1[0][1];
    const w12 = w1[0][2];
    const dL_dw10_vec = data.map((d, idx) => {
      const z0 = w10 * d.x0 + w11 * d.x1 + w12 * d.x2;
      const reluPrime = z0 > 0 ? 1 : 0;
      return (yhatVec[idx] - yVec[idx]) * w20 * reluPrime * d.x0;
    });
    q9El.innerHTML = `[${dL_dw10_vec.map(v=>v.toFixed(6)).join(', ')}]`;
  }
  // Q9: Full gradient matrices (averaged over dataset)
  if (q10W2El || q10W1El) {
    const N = data.length;
    // dL/dw2 (1x4): average over samples of (yhat - y) * h_j for each j
    const gradW2 = new Array(4).fill(0);
    for (let s = 0; s < N; s++) {
      const deltaOut = yhatVec[s] - yVec[s]; // since dL/dz_out = yhat - y
      for (let j = 0; j < 4; j++) gradW2[j] += deltaOut * hMat[s][j];
    }
    for (let j = 0; j < 4; j++) gradW2[j] /= N;

    // dL/dw1 (4x3): for each hidden j and input i: average of (dL/dH_j) * ReLU'(z_j) * x_i
    // where dL/dH_j = (yhat - y) * w2_j
    const gradW1 = Array.from({ length: 4 }, () => new Array(3).fill(0));
    for (let s = 0; s < N; s++) {
      // Compute z for each hidden unit j to get ReLU'
      for (let j = 0; j < 4; j++) {
        const z_j = w1[j][0] * data[s].x0 + w1[j][1] * data[s].x1 + w1[j][2] * data[s].x2;
        const reluPrime = z_j > 0 ? 1 : 0;
        const dL_dHj = (yhatVec[s] - yVec[s]) * w2[0][j];
        const factor = (dL_dHj) * reluPrime;
        // accumulate for inputs i = 0..2
        gradW1[j][0] += factor * data[s].x0;
        gradW1[j][1] += factor * data[s].x1;
        gradW1[j][2] += factor * data[s].x2;
      }
    }
    for (let j = 0; j < 4; j++) {
      for (let i = 0; i < 3; i++) gradW1[j][i] /= N;
    }

    if (q10W2El) {
      q10W2El.textContent = `[ ${gradW2.map(v=>v.toFixed(6)).join(', ')} ]`;
    }
    if (q10W1El) {
      // Pretty print 4x3 matrix as rows
      const rows = gradW1.map(row => `[ ${row.map(v=>v.toFixed(6)).join(', ')} ]`);
      q10W1El.textContent = `[\n  ${rows.join(',\n  ')}\n]`;
    }
    // If Q10 exists, compute updated weights using lr=0.23
    if (q11UpdW2El || q11UpdW1El) {
      const lr = 0.23;
      const newW2 = gradW2.map((g, j) => w2[0][j] - lr * g);
      const newW1 = gradW1.map((row, j) => row.map((g, i) => w1[j][i] - lr * g));
      if (q11UpdW2El) q11UpdW2El.textContent = `[ ${newW2.map(v=>v.toFixed(6)).join(', ')} ]`;
      if (q11UpdW1El) {
        const rowsNew = newW1.map(row => `[ ${row.map(v=>v.toFixed(6)).join(', ')} ]`);
        q11UpdW1El.textContent = `[\n  ${rowsNew.join(',\n  ')}\n]`;
      }
    }
  }
}


