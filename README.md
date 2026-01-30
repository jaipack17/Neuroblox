# Neural Network Builder

A visual drag-and-drop interface for building neural networks with automatic PyTorch code generation. Built with Electron.js and Google Blockly.

## Features

- 🎨 **Visual Block Programming**: Drag and drop blocks to build neural networks
- 🧠 **Neural Network Components**: 
  - Layers: Dense, Conv2D, MaxPool2D, Flatten, Dropout
  - Activations: ReLU, Sigmoid, Tanh, Softmax
  - Training: Optimizers, Loss Functions, Training Configuration
- 🔄 **Real-time Code Generation**: Automatically generates PyTorch code as you build
- 📋 **Export & Copy**: Export generated code as Python files or copy to clipboard
- 🎯 **Production Ready**: Clean, modern UI with dark theme

## Installation

1. Install dependencies:
```bash
npm install
```

2. Run the application:
```bash
npm start
```

For development with DevTools:
```bash
npm run dev
```

## Usage

1. **Add Blocks**: Click on blocks in the left sidebar to add them to the workspace
2. **Connect Blocks**: Drag blocks to connect them in sequence
3. **Configure**: Click on blocks to edit their parameters (units, kernel size, etc.)
4. **View Code**: Generated PyTorch code appears in the right panel in real-time
5. **Export**: Click "Export PyTorch Code" to save the code as a Python file

## Project Structure

```
frontend/
├── main.js              # Electron main process
├── index.html           # Main UI
├── styles.css           # Styling
├── app.js               # Application logic
├── custom-blocks.js     # Blockly custom block definitions
├── pytorch-generator.js # PyTorch code generator
└── package.json         # Dependencies
```

## Customization

### Adding New Blocks

Edit `custom-blocks.js` to add new block types. Then update `pytorch-generator.js` to handle code generation for the new blocks.

### Styling

Modify `styles.css` to customize the appearance. The app uses CSS variables for easy theming.

## Generated Code Format

The generator creates a complete PyTorch script with:
- Model class definition (nn.Module)
- Sequential layer architecture
- Optimizer configuration
- Loss function
- Training loop template

## License

MIT

