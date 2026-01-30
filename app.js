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

// Update generated code
function updateCode() {
  if (!workspace) return;

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

// Switch view between PyTorch and JSON
function switchView(view) {
  currentView = view;
  
  // Update button states
  document.getElementById('pytorchViewBtn').classList.toggle('active', view === 'pytorch');
  document.getElementById('jsonViewBtn').classList.toggle('active', view === 'json');
  
  // Update code
  updateCode();
}

// Send JSON to backend
async function sendToBackend() {
  if (!workspace || !jsonGenerator) {
    showNotification('No data to send', 'error');
    return;
  }

  try {
    const json = jsonGenerator.generate(workspace);
    const jsonData = JSON.parse(json);
    
    // Get backend URL (you can configure this)
    const backendUrl = prompt('Enter backend API URL:', 'http://localhost:8000/api/train') || 'http://localhost:8000/api/train';
    
    showNotification('Sending to backend...', 'success');
    
    const response = await fetch(backendUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: json
    });

    if (response.ok) {
      const result = await response.json();
      showNotification('Successfully sent to backend!');
      console.log('Backend response:', result);
    } else {
      const error = await response.text();
      showNotification(`Backend error: ${response.status}`, 'error');
      console.error('Backend error:', error);
    }
  } catch (error) {
    console.error('Error sending to backend:', error);
    showNotification(`Error: ${error.message}`, 'error');
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initBlockly();

  // Setup button handlers
  document.getElementById('exportBtn').addEventListener('click', exportCode);
  document.getElementById('copyBtn').addEventListener('click', copyCode);
  document.getElementById('clearBtn').addEventListener('click', clearWorkspace);
  document.getElementById('sendBtn').addEventListener('click', sendToBackend);
  
  // Setup view toggle
  document.getElementById('pytorchViewBtn').addEventListener('click', () => switchView('pytorch'));
  document.getElementById('jsonViewBtn').addEventListener('click', () => switchView('json'));
});
