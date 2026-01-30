// PyTorch Code Generator for Blockly

class PyTorchGenerator {
  constructor() {
    this.imports = new Set();
    this.layers = [];
    this.layerSizes = [];
    this.inputSize = 784;
    this.lossFunction = null;
  }

  generate(workspace) {
    // Reset state
    this.imports.clear();
    this.layers = [];
    this.layerSizes = [];
    this.inputSize = 784;
    this.lossFunction = null;

    // Get all top-level blocks (blocks with no previous connection)
    const allBlocks = workspace.getAllBlocks(false);
    const topBlocks = allBlocks.filter(block => {
      // A top block is one that has no previous connection
      return !block.getPreviousBlock() && block.type !== 'cross_entropy_loss';
    });

    // Separate loss blocks
    const lossBlocks = allBlocks.filter(block => 
      block.type === 'cross_entropy_loss'
    );

    // Process each chain of connected blocks sequentially
    for (let startBlock of topBlocks) {
      this.processBlockChain(startBlock);
    }

    // Process loss blocks (they don't need to be connected)
    for (let block of lossBlocks) {
      this.processBlock(block);
    }

    return this.generateCode();
  }

  processBlockChain(block) {
    // Process blocks in sequence following connections
    let currentBlock = block;
    while (currentBlock) {
      this.processBlock(currentBlock);
      currentBlock = currentBlock.getNextBlock();
    }
  }

  processBlock(block) {
    if (!block) return;

    const blockType = block.type;

    switch(blockType) {
      case 'input_layer':
        this.processInputLayer(block);
        break;
      case 'dense_layer':
        this.processDenseLayer(block);
        break;
      case 'relu_activation':
        this.processReLUActivation(block);
        break;
      case 'sigmoid_activation':
        this.processSigmoidActivation(block);
        break;
      case 'dropout_layer':
        this.processDropoutLayer(block);
        break;
      case 'cross_entropy_loss':
        this.processCrossEntropyLoss(block);
        break;
    }
  }

  processInputLayer(block) {
    this.inputSize = parseInt(block.getFieldValue('input_size')) || 784;
  }

  processDenseLayer(block) {
    this.imports.add('torch.nn');
    const units = parseInt(block.getFieldValue('units')) || 128;
    
    // Track layer sizes
    if (!this.layerSizes) {
      this.layerSizes = [];
    }
    
    // For the first dense layer, use input_size
    if (this.layerSizes.length === 0) {
      this.layers.push(`nn.Linear(${this.inputSize}, ${units})`);
      this.layerSizes.push({ input: this.inputSize, output: units });
    } else {
      // Use previous layer's output as this layer's input
      const prevOutput = this.layerSizes[this.layerSizes.length - 1].output;
      this.layers.push(`nn.Linear(${prevOutput}, ${units})`);
      this.layerSizes.push({ input: prevOutput, output: units });
    }
  }

  processReLUActivation(block) {
    this.imports.add('torch.nn');
    this.layers.push(`nn.ReLU()`);
  }

  processSigmoidActivation(block) {
    this.imports.add('torch.nn');
    this.layers.push(`nn.Sigmoid()`);
  }

  processDropoutLayer(block) {
    this.imports.add('torch.nn');
    const prob = parseFloat(block.getFieldValue('prob')) || 0.2;
    this.layers.push(`nn.Dropout(${prob})`);
  }

  processCrossEntropyLoss(block) {
    this.imports.add('torch.nn');
    this.lossFunction = 'CrossEntropyLoss';
  }

  generateCode() {
    let code = '';

    // Imports
    code += 'import torch\n';
    code += 'import torch.nn as nn\n';
    code += 'import torch.optim as optim\n';
    code += 'from torch.utils.data import DataLoader\n\n';

    // Model Definition
    code += 'class NeuralNetwork(nn.Module):\n';
    code += '    def __init__(self):\n';
    code += '        super(NeuralNetwork, self).__init__()\n';
    
    if (this.layers.length > 0) {
      code += '        self.layers = nn.Sequential(\n';
      this.layers.forEach((layer, index) => {
        const comma = index < this.layers.length - 1 ? ',' : '';
        code += `            ${layer}${comma}\n`;
      });
      code += '        )\n';
    } else {
      code += '        # Add layers here\n';
      code += '        pass\n';
    }
    
    code += '\n';
    code += '    def forward(self, x):\n';
    if (this.layers.length > 0) {
      code += '        return self.layers(x)\n';
    } else {
      code += '        # Implement forward pass\n';
      code += '        return x\n';
    }

    code += '\n\n';
    code += '# Initialize model\n';
    code += 'model = NeuralNetwork()\n\n';

    // Optimizer (default)
    code += `# Optimizer\n`;
    code += `optimizer = optim.Adam(model.parameters(), lr=0.001)\n\n`;

    // Loss Function
    if (this.lossFunction) {
      code += `# Loss Function\n`;
      code += `criterion = nn.${this.lossFunction}()\n\n`;
    } else {
      code += `# Loss Function (default: CrossEntropyLoss)\n`;
      code += `criterion = nn.CrossEntropyLoss()\n\n`;
    }

    // Training Loop Template
    code += `# Training Configuration\n`;
    code += `epochs = 10\n`;
    code += `batch_size = 32\n\n`;

    code += `# Training Loop\n`;
    code += `def train_model(model, train_loader, optimizer, criterion, epochs):\n`;
    code += `    model.train()\n`;
    code += `    for epoch in range(epochs):\n`;
    code += `        running_loss = 0.0\n`;
    code += `        for batch_idx, (data, target) in enumerate(train_loader):\n`;
    code += `            # Zero gradients\n`;
    code += `            optimizer.zero_grad()\n`;
    code += `            \n`;
    code += `            # Forward pass\n`;
    code += `            output = model(data)\n`;
    code += `            \n`;
    code += `            # Calculate loss\n`;
    code += `            loss = criterion(output, target)\n`;
    code += `            \n`;
    code += `            # Backward pass\n`;
    code += `            loss.backward()\n`;
    code += `            \n`;
    code += `            # Update weights\n`;
    code += `            optimizer.step()\n`;
    code += `            \n`;
    code += `            running_loss += loss.item()\n`;
    code += `        \n`;
    code += `        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')\n`;
    code += `    \n`;
    code += `    return model\n\n`;

    code += `# Example usage:\n`;
    code += `# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n`;
    code += `# trained_model = train_model(model, train_loader, optimizer, criterion, epochs)\n`;

    return code;
  }
}
