// Main Application Logic

let workspace;
let generator;
let jsonGenerator;
let currentView = 'pytorch'; // 'pytorch' or 'json'
let draggedBlockType = null;

// Initialize Blockly workspace
function initBlockly() {
  const blocklyDiv = document.getElementById('blocklyDiv');
  
  workspace = Blockly.inject(blocklyDiv, {
    toolbox: {
      kind: 'categoryToolbox',
      contents: []
    },
    grid: {
      spacing: 20,
      length: 3,
      colour: '#e5e7eb',
      snap: true
    },
    zoom: {
      controls: true,
      wheel: true,
      startScale: 1.0,
      maxScale: 3,
      minScale: 0.3,
      scaleSpeed: 1.2
    },
    trashcan: true,
    media: 'https://unpkg.com/blockly@11.0.0/media/',
    theme: Blockly.Themes.Classic,
    move: {
      scrollbars: true,
      drag: true,
      wheel: true
    }
  });

  // Initialize generators
  generator = new PyTorchGenerator();
  jsonGenerator = new JSONGenerator();

  // Listen for workspace changes - only update code on meaningful changes
  workspace.addChangeListener((event) => {
    // Only update code on block connections/disconnections, not on every change
    if (event.type === Blockly.Events.BLOCK_CREATE ||
        event.type === Blockly.Events.BLOCK_DELETE) {
      setTimeout(updateCode, 100);
    } else if (event.type === Blockly.Events.BLOCK_CHANGE) {
      // Update for any block change (including dataset, training config)
      setTimeout(updateCode, 100);
    } else if (event.type === Blockly.Events.BLOCK_MOVE) {
      // Only update if blocks are being connected/disconnected
      const moveEvent = event;
      if (moveEvent.newParentId || moveEvent.oldParentId || 
          moveEvent.newInputName || moveEvent.oldInputName) {
        setTimeout(updateCode, 100);
      }
    }
  });

  // Setup drag and drop from sidebar to workspace
  setupDragAndDrop();

  // Make workspace responsive
  window.addEventListener('resize', onResize);
  onResize();
}

// Setup drag and drop functionality
function setupDragAndDrop() {
  const paletteBlocks = document.querySelectorAll('.palette-block');
  const workspaceDiv = document.getElementById('blocklyDiv');

  paletteBlocks.forEach(block => {
    // Drag start
    block.addEventListener('dragstart', (e) => {
      draggedBlockType = block.getAttribute('data-block-type');
      e.dataTransfer.effectAllowed = 'copy';
      block.style.opacity = '0.5';
    });

    // Drag end
    block.addEventListener('dragend', (e) => {
      block.style.opacity = '1';
      draggedBlockType = null;
    });
  });

  // Allow drop on workspace
  workspaceDiv.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
  });

  // Handle drop on workspace
  workspaceDiv.addEventListener('drop', (e) => {
    e.preventDefault();
    if (draggedBlockType) {
      // Get drop coordinates relative to workspace
      const rect = workspaceDiv.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      // Convert screen coordinates to workspace coordinates
      const metrics = workspace.getMetrics();
      const scrollX = workspace.scrollX || 0;
      const scrollY = workspace.scrollY || 0;
      
      // Account for scroll and convert to workspace coordinates
      const workspaceX = scrollX + x;
      const workspaceY = scrollY + y;
      
      createBlockAtPosition(draggedBlockType, workspaceX, workspaceY);
      draggedBlockType = null;
    }
  });
}

// Create block at specific position
function createBlockAtPosition(blockType, x, y) {
  if (!workspace) return;

  let blockDefinition = null;

  switch(blockType) {
    case 'input':
      blockDefinition = workspace.newBlock('input_layer');
      break;
    case 'dense':
      blockDefinition = workspace.newBlock('dense_layer');
      break;
    case 'relu':
      blockDefinition = workspace.newBlock('relu_activation');
      break;
    case 'sigmoid':
      blockDefinition = workspace.newBlock('sigmoid_activation');
      break;
    case 'softmax':
      blockDefinition = workspace.newBlock('softmax_activation');
      break;
    case 'cross_entropy':
      blockDefinition = workspace.newBlock('cross_entropy_loss');
      break;
    case 'dropout':
      blockDefinition = workspace.newBlock('dropout_layer');
      break;
    case 'dataset':
      blockDefinition = workspace.newBlock('dataset');
      break;
    case 'training_config':
      blockDefinition = workspace.newBlock('training_config');
      break;
  }

  if (blockDefinition) {
    // Position block at drop location
    blockDefinition.moveBy(x - 100, y - 50);
    
    // Initialize and render
    blockDefinition.initSvg();
    blockDefinition.render();
    
    // Render workspace
    workspace.render();
  }
}

// Handle window resize
function onResize() {
  if (workspace) {
    Blockly.svgResize(workspace);
  }
}

// Update generated code or diagram
function updateCode() {
  if (!workspace) return;
  if (currentView === 'diagram') {
    updateDiagram();
    return;
  }
  try {
    const codeOutput = document.getElementById('codeOutput');
    
    if (currentView === 'json') {
      if (!jsonGenerator) return;
      
      // Check if there are any blocks
      const allBlocks = workspace.getAllBlocks(false);
      if (allBlocks.length === 0) {
        codeOutput.querySelector('code').textContent = '# Drag blocks from the sidebar to generate JSON...';
        return;
      }
      
      const json = jsonGenerator.generate(workspace);
      codeOutput.querySelector('code').textContent = json;
    } else {
      // PyTorch view
      if (!generator) return;
      
      // Check if there are any connected blocks
      const allBlocks = workspace.getAllBlocks(false);
      const hasConnectedBlocks = allBlocks.some(block => 
        block.getPreviousBlock() || block.getNextBlock()
      );
      
      // Check for loss blocks
      const hasLossBlocks = allBlocks.some(block => 
        block.type === 'cross_entropy_loss'
      );
      
      if (!hasConnectedBlocks && !hasLossBlocks) {
        codeOutput.querySelector('code').textContent = '# Connect blocks to generate code...\n# Drag blocks from the sidebar to the workspace, then connect them together.';
        return;
      }
      
      const code = generator.generate(workspace);
      codeOutput.querySelector('code').textContent = code;
    }
  } catch (error) {
    console.error('Error generating code:', error);
    const codeOutput = document.getElementById('codeOutput');
    codeOutput.querySelector('code').textContent = `# Error generating code: ${error.message}`;
  }
}

// Export code
function exportCode() {
  if (!workspace || !generator) return;

  try {
    const code = generator.generate(workspace);
    
    // Create blob and download
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'neural_network.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showNotification('Code exported successfully!');
  } catch (error) {
    console.error('Error exporting code:', error);
    showNotification('Error exporting code', 'error');
  }
}

// Copy code to clipboard
function copyCode() {
  const codeOutput = document.getElementById('codeOutput');
  const code = codeOutput.querySelector('code').textContent;
  
  navigator.clipboard.writeText(code).then(() => {
    showNotification('Code copied to clipboard!');
  }).catch(err => {
    console.error('Failed to copy:', err);
    showNotification('Failed to copy code', 'error');
  });
}

// Clear workspace
function clearWorkspace() {
  if (!workspace) return;
  
  if (confirm('Are you sure you want to clear all blocks?')) {
    workspace.clear();
    updateCode();
  }
}

// Show notification
function showNotification(message, type = 'success') {
  const notification = document.createElement('div');
  notification.className = `notification ${type === 'error' ? 'error' : ''}`;
  notification.textContent = message;
  document.body.appendChild(notification);

  setTimeout(() => {
    notification.style.animation = 'slideIn 0.3s ease reverse';
    setTimeout(() => {
      if (document.body.contains(notification)) {
        document.body.removeChild(notification);
      }
    }, 300);
  }, 2000);
}

// Sync the selected file path into all Dataset blocks so the block shows the same path
function syncSelectedPathToDatasetBlocks(path) {
  if (!workspace) return;
  const allBlocks = workspace.getAllBlocks(false);
  allBlocks.forEach(block => {
    if (block.type === 'dataset') {
      block.setFieldValue(path, 'path');
    }
  });
  workspace.render();
}

// Switch view between PyTorch and JSON (Diagram opens full-screen overlay)
function switchView(view) {
  currentView = view;
  document.getElementById('pytorchViewBtn').classList.toggle('active', view === 'pytorch');
  document.getElementById('jsonViewBtn').classList.toggle('active', view === 'json');
  const diagramBtn = document.getElementById('diagramViewBtn');
  if (diagramBtn) diagramBtn.classList.toggle('active', view === 'diagram');
  if (view === 'diagram') {
    openDiagramFullScreen();
    return;
  }
  updateCode();
}

// Build ordered list of layers from the first connected chain (for diagram)
function getLayerSequence(workspace) {
  if (!workspace) return [];
  const allBlocks = workspace.getAllBlocks(false);
  const topBlocks = allBlocks.filter(block =>
    !block.getPreviousBlock() &&
    block.type !== 'dataset' &&
    block.type !== 'training_config'
  );
  const sequence = [];
  const startBlock = topBlocks[0];
  if (!startBlock) return [];
  let block = startBlock;
  while (block) {
    const item = layerToDiagramItem(block);
    if (item) sequence.push(item);
    block = block.getNextBlock();
  }
  return sequence;
}

function layerToDiagramItem(block) {
  switch (block.type) {
    case 'input_layer': {
      const size = parseInt(block.getFieldValue('input_size'), 10) || 784;
      return { type: 'input', label: 'Input', size };
    }
    case 'dense_layer': {
      const size = parseInt(block.getFieldValue('units'), 10) || 128;
      return { type: 'dense', label: 'Dense', size };
    }
    case 'relu_activation':
      return { type: 'relu', label: 'ReLU' };
    case 'sigmoid_activation':
      return { type: 'sigmoid', label: 'Sigmoid' };
    case 'softmax_activation':
      return { type: 'softmax', label: 'Softmax' };
    case 'dropout_layer': {
      const prob = parseFloat(block.getFieldValue('prob')) || 0.2;
      return { type: 'dropout', label: 'Dropout', prob };
    }
    default:
      return null;
  }
}

// --- Full-screen interactive diagram (3B1B style, pan/zoom, dropout) ---

const DIAGRAM_MAX_NEURONS = 25;
const DIAGRAM_CONTENT_PADDING = 70;
const DIAGRAM_COLUMN_GAP = 100;
const DIAGRAM_NODE_RADIUS = 6;
const DIAGRAM_NODE_SPACING = 22;
const DIAGRAM_LINE_OPACITY = 0.15;
const DIAGRAM_FIT_PADDING = 0.82;

let diagramState = {
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  isPanning: false,
  lastX: 0,
  lastY: 0,
  contentWidth: 0,
  contentHeight: 0,
  columnData: null
};

function getDiagramColumns(sequence) {
  const columns = [];
  let activationLabel = null;
  for (const layer of sequence) {
    if (layer.size != null) {
      columns.push({
        size: layer.size,
        type: layer.type,
        activationBefore: activationLabel,
        dropoutAfter: null
      });
      activationLabel = null;
    } else {
      if (layer.type === 'dropout' && layer.prob != null && columns.length > 0) {
        columns[columns.length - 1].dropoutAfter = layer.prob;
      }
      activationLabel = layer.label + (layer.prob != null && layer.type !== 'dropout' ? ` ${layer.prob}` : '');
    }
  }
  return columns;
}

function getInactiveNeuronIndices(n, dropoutProb, _columnIndex) {
  if (dropoutProb <= 0 || dropoutProb >= 1) return new Set();
  const k = Math.max(0, Math.min(n, Math.floor(n * dropoutProb)));
  if (k <= 0) return new Set();
  const indices = new Set();
  const step = n / Math.max(1, k);
  for (let i = 0; i < k; i++) {
    indices.add(Math.min(Math.floor(i * step), n - 1));
  }
  return indices;
}

function buildDiagramContent(columns) {
  const colCount = columns.length;
  const maxN = Math.max(1, ...columns.map(c => Math.min(c.size, DIAGRAM_MAX_NEURONS)));
  const contentWidth = 2 * DIAGRAM_CONTENT_PADDING + (colCount - 1) * DIAGRAM_COLUMN_GAP + colCount * (DIAGRAM_NODE_RADIUS * 4);
  const contentHeight = 2 * DIAGRAM_CONTENT_PADDING + maxN * DIAGRAM_NODE_SPACING;

  const columnData = columns.map((col, i) => {
    const n = Math.min(col.size, DIAGRAM_MAX_NEURONS);
    const cx = DIAGRAM_CONTENT_PADDING + DIAGRAM_NODE_RADIUS * 2 + i * (DIAGRAM_COLUMN_GAP + DIAGRAM_NODE_RADIUS * 4);
    const spacing = n <= 1 ? 0 : (contentHeight - 2 * DIAGRAM_CONTENT_PADDING) / (n + 1);
    const ys = Array.from({ length: n }, (_, j) => DIAGRAM_CONTENT_PADDING + spacing * (j + 1));
    const inactive = col.dropoutAfter != null
      ? getInactiveNeuronIndices(n, col.dropoutAfter, i)
      : new Set();
    return {
      cx,
      ys,
      size: col.size,
      type: col.type,
      activationBefore: col.activationBefore,
      inactive
    };
  });

  return { columnData, contentWidth, contentHeight };
}

function drawDiagram(ctx, columnData, contentWidth, contentHeight, scale, offsetX, offsetY, viewportW, viewportH) {
  ctx.save();
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, viewportW, viewportH);

  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  const contentLeft = 0;
  const contentTop = 0;
  ctx.fillStyle = '#161b22';
  ctx.fillRect(contentLeft, contentTop, contentWidth, contentHeight);

  ctx.lineWidth = 1 / scale;
  for (let c = 0; c < columnData.length - 1; c++) {
    const curr = columnData[c];
    const next = columnData[c + 1];
    for (let i = 0; i < curr.ys.length; i++) {
      for (let j = 0; j < next.ys.length; j++) {
        const fromInactive = curr.inactive && curr.inactive.has(i);
        const toInactive = next.inactive && next.inactive.has(j);
        ctx.strokeStyle = fromInactive || toInactive
          ? 'rgba(80, 80, 80, 0.06)'
          : 'rgba(0, 212, 255, ' + DIAGRAM_LINE_OPACITY + ')';
        ctx.beginPath();
        ctx.moveTo(curr.cx, curr.ys[i]);
        ctx.lineTo(next.cx, next.ys[j]);
        ctx.stroke();
      }
    }
  }

  ctx.strokeStyle = 'rgba(0, 212, 255, 0.25)';
  ctx.lineWidth = 1.5 / scale;
  for (const col of columnData) {
    for (let i = 0; i < col.ys.length; i++) {
      const inactive = col.inactive && col.inactive.has(i);
      ctx.beginPath();
      ctx.arc(col.cx, col.ys[i], DIAGRAM_NODE_RADIUS, 0, Math.PI * 2);
      if (inactive) {
        ctx.fillStyle = '#30363d';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      } else {
        ctx.fillStyle = '#00d4ff';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      }
      ctx.stroke();
    }
  }

  ctx.restore();
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
  ctx.font = Math.max(10, 12 / scale) + 'px sans-serif';
  ctx.textAlign = 'center';
  const labelY = contentHeight - 18;
  for (let c = 0; c < columnData.length; c++) {
    const col = columnData[c];
    const sizeLabel = col.type === 'input'
      ? (col.size > DIAGRAM_MAX_NEURONS ? `Input (${col.size})` : 'Input')
      : (col.size > DIAGRAM_MAX_NEURONS ? `${col.size}` : '');
    if (sizeLabel) ctx.fillText(sizeLabel, col.cx, labelY);
    if (col.activationBefore && c > 0) {
      const midX = (columnData[c - 1].cx + col.cx) / 2;
      ctx.fillText(col.activationBefore, midX, contentHeight / 2);
    }
  }
  ctx.restore();
}

function openDiagramFullScreen() {
  const overlay = document.getElementById('diagramOverlay');
  const emptyEl = document.getElementById('diagramEmptyOverlay');
  const wrap = document.getElementById('diagramCanvasWrap');
  const canvas = document.getElementById('diagramCanvas');
  if (!overlay || !wrap || !canvas) return;

  const sequence = getLayerSequence(workspace);
  const columns = getDiagramColumns(sequence);
  if (columns.length === 0) {
    overlay.classList.remove('hidden');
    emptyEl.classList.remove('hidden');
    wrap.classList.add('hidden');
    return;
  }
  emptyEl.classList.add('hidden');
  wrap.classList.remove('hidden');

  const { columnData, contentWidth, contentHeight } = buildDiagramContent(columns);
  diagramState.columnData = columnData;
  diagramState.contentWidth = contentWidth;
  diagramState.contentHeight = contentHeight;

  overlay.classList.remove('hidden');

  requestAnimationFrame(() => {
    const rect = wrap.getBoundingClientRect();
    const w = Math.max(rect.width, 400);
    const h = Math.max(rect.height, 300);
    const fitScale = Math.min(w / contentWidth, h / contentHeight) * DIAGRAM_FIT_PADDING;
    diagramState.scale = fitScale;
    diagramState.offsetX = (w - contentWidth * fitScale) / 2;
    diagramState.offsetY = (h - contentHeight * fitScale) / 2;
    resizeDiagramCanvas();
    redrawDiagram();
  });
  setupDiagramInteraction();
}

function resizeDiagramCanvas() {
  const wrap = document.getElementById('diagramCanvasWrap');
  const canvas = document.getElementById('diagramCanvas');
  if (!wrap || !canvas || !diagramState.columnData) return;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}

function redrawDiagram() {
  const wrap = document.getElementById('diagramCanvasWrap');
  const canvas = document.getElementById('diagramCanvas');
  if (!wrap || !canvas || !diagramState.columnData) return;
  const ctx = canvas.getContext('2d');
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  drawDiagram(
    ctx,
    diagramState.columnData,
    diagramState.contentWidth,
    diagramState.contentHeight,
    diagramState.scale,
    diagramState.offsetX,
    diagramState.offsetY,
    w,
    h
  );
}

function setupDiagramInteraction() {
  const wrap = document.getElementById('diagramCanvasWrap');
  const canvas = document.getElementById('diagramCanvas');
  if (!wrap || !canvas) return;

  const onWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    const newScale = Math.max(0.2, Math.min(4, diagramState.scale * (1 + delta)));
    const rect = wrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const wx = (mx - diagramState.offsetX) / diagramState.scale;
    const wy = (my - diagramState.offsetY) / diagramState.scale;
    diagramState.scale = newScale;
    diagramState.offsetX = mx - wx * newScale;
    diagramState.offsetY = my - wy * newScale;
    redrawDiagram();
  };

  const onDown = (e) => {
    if (e.button !== 0) return;
    diagramState.isPanning = true;
    diagramState.lastX = e.clientX;
    diagramState.lastY = e.clientY;
  };

  const onMove = (e) => {
    if (!diagramState.isPanning) return;
    diagramState.offsetX += e.clientX - diagramState.lastX;
    diagramState.offsetY += e.clientY - diagramState.lastY;
    diagramState.lastX = e.clientX;
    diagramState.lastY = e.clientY;
    redrawDiagram();
  };

  const onUp = () => {
    diagramState.isPanning = false;
  };

  wrap.removeEventListener('wheel', onWheel, { passive: false });
  wrap.addEventListener('wheel', onWheel, { passive: false });
  canvas.removeEventListener('mousedown', onDown);
  canvas.addEventListener('mousedown', onDown);
  window.removeEventListener('mousemove', onMove);
  window.addEventListener('mousemove', onMove);
  window.removeEventListener('mouseup', onUp);
  window.addEventListener('mouseup', onUp);
}

function closeDiagramOverlay() {
  document.getElementById('diagramOverlay').classList.add('hidden');
  diagramState.columnData = null;
}

function updateDiagram() {
  if (document.getElementById('diagramOverlay') && !document.getElementById('diagramOverlay').classList.contains('hidden')) {
    const sequence = getLayerSequence(workspace);
    const columns = getDiagramColumns(sequence);
    if (columns.length > 0) {
      const { columnData, contentWidth, contentHeight } = buildDiagramContent(columns);
      diagramState.columnData = columnData;
      diagramState.contentWidth = contentWidth;
      diagramState.contentHeight = contentHeight;
      resizeDiagramCanvas();
      redrawDiagram();
    }
  }
}

// --- Training screen & backend integration ---

let trainingPollId = null;
let currentJobId = null;
let lastEpoch = 0;
let lastTotalEpochs = 0;
const lossHistory = [];
const POLL_INTERVAL_MS = 1000;

function showCodeView() {
  document.getElementById('codeView').classList.remove('hidden');
  document.getElementById('trainingView').classList.add('hidden');
}

function showTrainingView() {
  document.getElementById('codeView').classList.add('hidden');
  document.getElementById('trainingView').classList.remove('hidden');
  document.getElementById('trainingError').classList.add('hidden');
  document.getElementById('trainingDone').classList.add('hidden');
  const overfitEl = document.getElementById('overfittingWarning');
  if (overfitEl) overfitEl.classList.add('hidden');
  lastEpoch = 0;
  lastTotalEpochs = 0;
  lossHistory.length = 0;
  drawLossChart();
}

function updateTrainingUI(state) {
  const statusEl = document.getElementById('trainingStatusLabel');
  const epochEl = document.getElementById('epochProgress');
  const progressBar = document.getElementById('progressBar');
  const progressPercent = document.getElementById('progressPercent');
  const lossEl = document.getElementById('lossValue');
  const trainBtn = document.getElementById('trainBtn');

  const status = state.status || 'idle';
  const isTraining = status === 'training';
  if (isTraining && state.total_epochs != null) lastTotalEpochs = state.total_epochs;
  if (isTraining && state.epoch != null) lastEpoch = state.epoch;
  const totalEpochs = state.total_epochs != null ? state.total_epochs : lastTotalEpochs;
  const epoch = state.epoch != null ? state.epoch : (status === 'complete' ? lastTotalEpochs : lastEpoch);
  const progress = isTraining && totalEpochs > 0
    ? (epoch / totalEpochs) * 100
    : status === 'complete'
      ? 100
      : 0;
  const loss = status === 'complete'
    ? state.final_loss
    : state.loss;

  statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);
  epochEl.textContent = `${epoch} / ${totalEpochs}`;
  progressBar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
  progressPercent.textContent = `${progress.toFixed(1)}%`;
  lossEl.textContent = typeof loss === 'number' ? loss.toFixed(4) : '—';

  const valAccEl = document.getElementById('valAccuracyValue');
  if (valAccEl) {
    const valAcc = state.val_accuracy;
    valAccEl.textContent = typeof valAcc === 'number' ? valAcc.toFixed(1) + '%' : '—';
  }

  trainBtn.disabled = isTraining;
  if (isTraining) {
    lossHistory.push({
      progress,
      loss: state.loss,
      val_loss: state.val_loss,
      val_accuracy: state.val_accuracy
    });
    drawLossChart();
    checkOverfitting();
  }

  if (status === 'failed' && state.error) {
    document.getElementById('trainingError').textContent = state.error;
    document.getElementById('trainingError').classList.remove('hidden');
  }

  if (status === 'complete' || status === 'failed') {
    stopPolling();
    const doneEl = document.getElementById('trainingDone');
    doneEl.classList.remove('hidden');
    doneEl.querySelector('.training-done-msg').textContent =
      status === 'complete'
        ? `Training complete. Final loss: ${typeof state.final_loss === 'number' ? state.final_loss.toFixed(4) : state.final_loss}`
        : `Training failed: ${state.error || 'Unknown'}`;
  }
}

function checkOverfitting() {
  const el = document.getElementById('overfittingWarning');
  if (!el || lossHistory.length < 3) return;
  const recent = lossHistory.slice(-5);
  const hasLoss = recent.every((d) => typeof d.loss === 'number');
  const hasValLoss = recent.every((d) => typeof d.val_loss === 'number');
  if (!hasLoss || !hasValLoss) return;
  const lossTrend = recent[recent.length - 1].loss < recent[0].loss;
  const valLossTrend = recent[recent.length - 1].val_loss > recent[0].val_loss;
  if (lossTrend && valLossTrend) {
    el.classList.remove('hidden');
  } else {
    el.classList.add('hidden');
  }
}

function drawLossChart() {
  const canvas = document.getElementById('lossChart');
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  if (w <= 0 || h <= 0) return;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  if (lossHistory.length === 0) {
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#9ca3af';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Loss will appear as training runs…', w / 2, h / 2);
    return;
  }

  ctx.fillStyle = '#f9fafb';
  ctx.fillRect(0, 0, w, h);

  const losses = lossHistory.map((d) => d.loss).filter((v) => typeof v === 'number' && isFinite(v));
  const valLosses = lossHistory.map((d) => d.val_loss).filter((v) => typeof v === 'number' && isFinite(v));
  const allVals = [...losses, ...valLosses];
  if (allVals.length < 1) return;

  const padding = { top: 12, right: 12, bottom: 24, left: 40 };
  const chartW = w - padding.left - padding.right;
  const chartH = h - padding.top - padding.bottom;
  const minL = Math.min(...allVals);
  const maxL = Math.max(...allVals);
  const range = maxL - minL || 1;
  const scaleY = (v) => padding.top + chartH - ((v - minL) / range) * chartH;
  const n = Math.max(losses.length, valLosses.length);
  const scaleX = (i, len) => padding.left + (len > 1 ? (i / (len - 1)) * chartW : chartW / 2);

  if (losses.length >= 1) {
    ctx.strokeStyle = '#6366f1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(scaleX(0, losses.length), scaleY(losses[0]));
    for (let i = 1; i < losses.length; i++) {
      ctx.lineTo(scaleX(i, losses.length), scaleY(losses[i]));
    }
    ctx.stroke();
  }
  if (valLosses.length >= 1) {
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(scaleX(0, valLosses.length), scaleY(valLosses[0]));
    for (let i = 1; i < valLosses.length; i++) {
      ctx.lineTo(scaleX(i, valLosses.length), scaleY(valLosses[i]));
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 1;
  ctx.font = '10px sans-serif';
  ctx.fillStyle = '#6b7280';
  const steps = 4;
  for (let i = 0; i <= steps; i++) {
    const y = padding.top + (chartH * (steps - i)) / steps;
    const val = minL + (range * i) / steps;
    ctx.fillText(val.toFixed(2), 4, y + 4);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
}

function startPolling(jobId) {
  if (trainingPollId || !jobId) return;
  currentJobId = jobId;
  function poll() {
    apiGetStatus(jobId)
      .then(updateTrainingUI)
      .catch((err) => {
        console.error('Status poll error:', err);
        document.getElementById('trainingStatusLabel').textContent = 'Connection error';
      });
  }
  poll();
  trainingPollId = setInterval(poll, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (trainingPollId) {
    clearInterval(trainingPollId);
    trainingPollId = null;
  }
  currentJobId = null;
}

async function startTraining() {
  if (!workspace || !jsonGenerator) {
    showNotification('Build a model and add Dataset + Training blocks', 'error');
    return;
  }

  const json = jsonGenerator.generate(workspace);
  let payload;
  try {
    payload = JSON.parse(json);
  } catch (e) {
    showNotification('Invalid model JSON', 'error');
    return;
  }

  if (!payload.blocks || payload.blocks.length === 0) {
    showNotification('Add at least one layer (e.g. Dense, ReLU) to the chain', 'error');
    return;
  }

  // Use file-picker path if set, else path from Dataset block
  const datasetPath = window.selectedDatasetPath || payload.dataset?.path;
  if (!payload.dataset) {
    showNotification('Add a Dataset block', 'error');
    return;
  }
  if (!datasetPath || String(datasetPath).trim() === '') {
    showNotification('Choose a dataset file (Browse) or set path in the Dataset block', 'error');
    return;
  }
  payload.dataset.path = datasetPath;
  const testSplit = window.testSplit != null ? Number(window.testSplit) : 0.2;
  payload.test_split = Math.max(0, Math.min(1, testSplit));

  try {
    const result = await apiStartTraining(payload);
    const jobId = result.job_id;
    if (jobId) {
      showTrainingView();
      updateTrainingUI({
        status: 'training',
        epoch: 0,
        total_epochs: payload.epochs,
        loss: 0
      });
      startPolling(jobId);
      showNotification('Training started');
    } else {
      showNotification('Backend did not return job_id', 'error');
    }
  } catch (err) {
    console.error(err);
    showNotification(err.message || 'Could not connect to backend. Is it running?', 'error');
  }
}

// --- Inference Playground ---
const INFERENCE_DEFAULT_INPUTS = 4;

function openInferencePlayground() {
  const overlay = document.getElementById('inferenceOverlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');
  loadInferenceModels();
  renderInferenceInputs(INFERENCE_DEFAULT_INPUTS);
  document.getElementById('inferenceMetadata').classList.add('hidden');
  document.getElementById('inferenceResult').classList.add('hidden');
  document.getElementById('inferenceModelSelect').value = '';
  document.getElementById('inferenceRunBtn').disabled = true;
}

function closeInferencePlayground() {
  document.getElementById('inferenceOverlay').classList.add('hidden');
}

async function loadInferenceModels() {
  const select = document.getElementById('inferenceModelSelect');
  if (!select) return;
  const first = select.options[0];
  select.innerHTML = '';
  select.appendChild(first);
  try {
    const data = await apiGetModels();
    const ids = Array.isArray(data) ? data : (data.models || data.job_ids || []);
    if (ids.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No trained models';
      select.appendChild(opt);
      return;
    }
    ids.forEach((id) => {
      const opt = document.createElement('option');
      opt.value = typeof id === 'string' ? id : (id.job_id || id.id || String(id));
      opt.textContent = 'Model #' + opt.value;
      select.appendChild(opt);
    });
  } catch (e) {
    console.error(e);
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'Failed to load models';
    select.appendChild(opt);
  }
}

function renderInferenceInputs(n) {
  const container = document.getElementById('inferenceInputs');
  if (!container) return;
  container.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const wrap = document.createElement('div');
    wrap.className = 'inference-input-row';
    const label = document.createElement('label');
    label.textContent = `Feature ${i + 1}`;
    label.className = 'inference-input-label';
    const input = document.createElement('input');
    input.type = 'number';
    input.step = 'any';
    input.className = 'inference-input-field';
    input.placeholder = '0';
    input.dataset.index = String(i);
    wrap.appendChild(label);
    wrap.appendChild(input);
    container.appendChild(wrap);
  }
}

function onInferenceModelSelect() {
  const select = document.getElementById('inferenceModelSelect');
  const meta = document.getElementById('inferenceMetadata');
  const runBtn = document.getElementById('inferenceRunBtn');
  const val = select && select.value;
  if (!val) {
    meta.classList.add('hidden');
    runBtn.disabled = true;
    return;
  }
  document.getElementById('inferenceJobId').textContent = val;
  document.getElementById('inferenceStatus').textContent = 'Ready';
  meta.classList.remove('hidden');
  runBtn.disabled = false;
}

async function runInference() {
  const select = document.getElementById('inferenceModelSelect');
  const jobId = select && select.value;
  if (!jobId) return;
  const inputs = document.querySelectorAll('.inference-input-field');
  const inputData = Array.from(inputs).map((inp) => parseFloat(inp.value) || 0);
  const resultEl = document.getElementById('inferenceResult');
  const predEl = document.getElementById('inferencePredClass');
  const rawEl = document.getElementById('inferenceRawOutput');
  if (!resultEl || !predEl || !rawEl) return;
  try {
    const result = await apiPredict(jobId, inputData);
    resultEl.classList.remove('hidden');
    predEl.textContent = result.prediction != null ? 'Class ' + result.prediction : (result.class != null ? 'Class ' + result.class : JSON.stringify(result));
    rawEl.textContent = result.probabilities ? JSON.stringify(result.probabilities) : (result.raw ? JSON.stringify(result.raw) : '—');
  } catch (e) {
    console.error(e);
    resultEl.classList.remove('hidden');
    predEl.textContent = 'Error';
    rawEl.textContent = e.message || String(e);
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initBlockly();

  document.getElementById('exportBtn').addEventListener('click', exportCode);
  document.getElementById('copyBtn').addEventListener('click', copyCode);
  document.getElementById('clearBtn').addEventListener('click', clearWorkspace);
  document.getElementById('trainBtn').addEventListener('click', startTraining);

  const testSplitSlider = document.getElementById('testSplitSlider');
  const testSplitBar = document.getElementById('testSplitBar');
  const testSplitLabel = document.getElementById('testSplitLabel');
  if (testSplitSlider && testSplitBar && testSplitLabel) {
    function updateTestSplit() {
      const val = parseFloat(testSplitSlider.value) || 0.2;
      window.testSplit = val;
      const trainPct = Math.round((1 - val) * 100);
      const testPct = Math.round(val * 100);
      testSplitBar.querySelector('.test-split-train').style.width = trainPct + '%';
      testSplitBar.querySelector('.test-split-test').style.width = testPct + '%';
      testSplitLabel.textContent = `${trainPct}% Training | ${testPct}% Testing`;
    }
    testSplitSlider.addEventListener('input', updateTestSplit);
    updateTestSplit();
  }

  const browseDatasetBtn = document.getElementById('browseDatasetBtn');
  const datasetPathDisplay = document.getElementById('datasetPathDisplay');
  browseDatasetBtn.addEventListener('click', async () => {
    const openPicker = window.electronAPI?.openDatasetFilePicker;
    if (typeof openPicker !== 'function') {
      showNotification('File picker is only available in the desktop app. Enter path in the Dataset block.', 'error');
      return;
    }
    try {
      const path = await openPicker();
      if (path != null && path !== '') {
        window.selectedDatasetPath = path;
        datasetPathDisplay.textContent = path;
        datasetPathDisplay.setAttribute('data-empty', 'false');
        datasetPathDisplay.title = path;
        syncSelectedPathToDatasetBlocks(path);
      }
    } catch (e) {
      console.error(e);
      showNotification('Could not open file picker', 'error');
    }
  });

  document.getElementById('backToBuilderBtn').addEventListener('click', () => {
    showCodeView();
    stopPolling();
    document.getElementById('trainBtn').disabled = false;
  });

  document.getElementById('pytorchViewBtn').addEventListener('click', () => switchView('pytorch'));
  document.getElementById('jsonViewBtn').addEventListener('click', () => switchView('json'));
  const diagramViewBtn = document.getElementById('diagramViewBtn');
  if (diagramViewBtn) diagramViewBtn.addEventListener('click', () => switchView('diagram'));

  const diagramBackBtn = document.getElementById('diagramBackBtn');
  if (diagramBackBtn) diagramBackBtn.addEventListener('click', () => {
    closeDiagramOverlay();
    currentView = 'pytorch';
    document.getElementById('pytorchViewBtn').classList.add('active');
    document.getElementById('jsonViewBtn').classList.remove('active');
    diagramViewBtn.classList.remove('active');
  });

  const inferenceBtn = document.getElementById('inferenceBtn');
  if (inferenceBtn) inferenceBtn.addEventListener('click', openInferencePlayground);
  const inferenceBackBtn = document.getElementById('inferenceBackBtn');
  if (inferenceBackBtn) inferenceBackBtn.addEventListener('click', closeInferencePlayground);
  const inferenceModelSelect = document.getElementById('inferenceModelSelect');
  if (inferenceModelSelect) inferenceModelSelect.addEventListener('change', onInferenceModelSelect);
  const inferenceRunBtn = document.getElementById('inferenceRunBtn');
  if (inferenceRunBtn) inferenceRunBtn.addEventListener('click', runInference);

  window.addEventListener('resize', () => {
    if (lossHistory.length > 0) drawLossChart();
    const overlay = document.getElementById('diagramOverlay');
    if (overlay && !overlay.classList.contains('hidden') && diagramState.columnData) {
      resizeDiagramCanvas();
      redrawDiagram();
    }
  });
});
