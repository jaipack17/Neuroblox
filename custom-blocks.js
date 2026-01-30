// Custom Blockly blocks for Neural Network Builder

// Input Layer Block
Blockly.Blocks['input_layer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Input Layer")
        .appendField("Size:")
        .appendField(new Blockly.FieldNumber(784, 1), "input_size");
    this.setNextStatement(true, null);
    this.setColour(230);
    this.setTooltip("Input layer - defines the input size");
    this.setHelpUrl("");
  }
};

// Dense/Fully Connected Layer Block
Blockly.Blocks['dense_layer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Fully Connected Layer")
        .appendField("Units:")
        .appendField(new Blockly.FieldNumber(128, 1), "units");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(200);
    this.setTooltip("Fully connected dense layer");
    this.setHelpUrl("");
  }
};

// ReLU Activation Block
Blockly.Blocks['relu_activation'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("ReLU Activation");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(120);
    this.setTooltip("Rectified Linear Unit activation");
    this.setHelpUrl("");
  }
};

// Sigmoid Activation Block
Blockly.Blocks['sigmoid_activation'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Sigmoid Activation");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(100);
    this.setTooltip("Sigmoid activation function");
    this.setHelpUrl("");
  }
};

// Cross Entropy Loss Block
Blockly.Blocks['cross_entropy_loss'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Cross Entropy Loss");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(20);
    this.setTooltip("Cross Entropy Loss function for classification");
    this.setHelpUrl("");
  }
};

// Dropout Layer Block
Blockly.Blocks['dropout_layer'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Dropout")
        .appendField("Probability:")
        .appendField(new Blockly.FieldNumber(0.2, 0, 1, 0.1), "prob");
    this.setPreviousStatement(true, null);
    this.setNextStatement(true, null);
    this.setColour(140);
    this.setTooltip("Dropout layer for regularization");
    this.setHelpUrl("");
  }
};

// Dataset Block
Blockly.Blocks['dataset'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Dataset")
        .appendField("Type:")
        .appendField(new Blockly.FieldDropdown([
          ["CSV", "csv"],
          ["Image", "image"],
          ["Text", "text"]
        ]), "dataset_type");
    this.appendDummyInput()
        .appendField("Path:")
        .appendField(new Blockly.FieldTextInput("data_examples/iris.csv"), "path");
    this.appendDummyInput()
        .appendField("Batch Size:")
        .appendField(new Blockly.FieldNumber(16, 1), "batch_size");
    this.setColour(40);
    this.setTooltip("Dataset configuration");
    this.setHelpUrl("");
  }
};

// Training Config Block
Blockly.Blocks['training_config'] = {
  init: function() {
    this.appendDummyInput()
        .appendField("Training Configuration");
    this.appendDummyInput()
        .appendField("Epochs:")
        .appendField(new Blockly.FieldNumber(50, 1), "epochs");
    this.setColour(10);
    this.setTooltip("Training hyperparameters");
    this.setHelpUrl("");
  }
};
