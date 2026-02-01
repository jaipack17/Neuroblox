// JSON Generator for Backend API

class JSONGenerator {
  constructor() {
    this.blocks = [];
    this.dataset = null;
    this.epochs = 50;
  }

  generate(workspace) {
    // Reset state
    this.blocks = [];
    this.dataset = null;
    this.epochs = 50;

    // Get all blocks in workspace
    const allBlocks = workspace.getAllBlocks(false);
    
    // Separate blocks by type
    const layerBlocks = [];
    const datasetBlocks = [];
    const trainingBlocks = [];

    // Get top-level blocks (blocks with no previous connection)
    const topBlocks = allBlocks.filter(block => {
      return !block.getPreviousBlock() && 
             block.type !== 'dataset' && 
             block.type !== 'training_config';
    });

    // Process layer chain
    for (let startBlock of topBlocks) {
      this.processBlockChain(startBlock);
    }

    // Process dataset and training config blocks (they don't need connections)
    allBlocks.forEach(block => {
      if (block.type === 'dataset') {
        this.processDataset(block);
      } else if (block.type === 'training_config') {
        this.processTrainingConfig(block);
      }
    });

    return this.generateJSON();
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
        // Input layer is not included in blocks array, just used for reference
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
      case 'softmax_activation':
        this.processSoftmaxActivation(block);
        break;
      case 'dropout_layer':
        this.processDropoutLayer(block);
        break;
    }
  }

  processDenseLayer(block) {
    const size = parseInt(block.getFieldValue('units')) || 128;
    this.blocks.push({
      type: 'dense',
      size: size
    });
  }

  processReLUActivation(block) {
    this.blocks.push({
      type: 'relu'
    });
  }

  processSigmoidActivation(block) {
    this.blocks.push({
      type: 'sigmoid'
    });
  }

  processSoftmaxActivation(block) {
    this.blocks.push({
      type: 'softmax'
    });
  }

  processDropoutLayer(block) {
    const prob = parseFloat(block.getFieldValue('prob')) || 0.2;
    this.blocks.push({
      type: 'dropout',
      prob: prob
    });
  }

  processDataset(block) {
    const datasetType = block.getFieldValue('dataset_type') || 'csv';
    const path = block.getFieldValue('path') || 'data_examples/iris.csv';
    const batchSize = parseInt(block.getFieldValue('batch_size')) || 16;
    
    this.dataset = {
      type: datasetType,
      path: path,
      batch_size: batchSize
    };
  }

  processTrainingConfig(block) {
    this.epochs = parseInt(block.getFieldValue('epochs')) || 50;
  }

  generateJSON() {
    const json = {
      epochs: this.epochs,
      dataset: this.dataset || {
        type: 'csv',
        path: 'data_examples/iris.csv',
        batch_size: 16
      },
      blocks: this.blocks
    };

    return JSON.stringify(json, null, 2);
  }
}
