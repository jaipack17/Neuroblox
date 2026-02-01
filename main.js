const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

// Store all python processes in an array or object
let pythonProcesses = [];

function getScriptPath(fileName) {
  // Helper to get the correct path in Dev vs Prod
  return app.isPackaged
    ? path.join(process.resourcesPath, fileName)
    : path.join(__dirname, 'resources', fileName);
}

function startPythonProcesses() {
  const scripts = ['server', 'generator'];

  scripts.forEach((scriptName) => {
    const scriptPath = getScriptPath(scriptName);
    console.log(`Starting ${scriptName} from: ${scriptPath}`);

    const proc = spawn(scriptPath);

    // Logging for debugging
    proc.stdout.on('data', (data) => console.log(`${scriptName}: ${data}`));
    proc.stderr.on('data', (data) => console.error(`${scriptName} Error: ${data}`));
    
    // Add to our list so we can kill it later
    pythonProcesses.push(proc);
  });
}

function stopPythonProcesses() {
  pythonProcesses.forEach((proc) => {
    console.log(`Killing process PID: ${proc.pid}`);
    proc.kill(); // Sends SIGTERM
  });
  pythonProcesses = []; // Clear the list
}

let mainWindow;

function createWindow() {
  const preloadPath = path.join(__dirname, 'preload.js');
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png'),
    titleBarStyle: 'default',
    backgroundColor: '#1e1e1e'
  });

  mainWindow.loadFile('index.html');

  // Open DevTools in development
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

ipcMain.handle('dialog:openDatasetFile', async () => {
  const { filePaths } = await dialog.showOpenDialog(mainWindow, {
    title: 'Choose dataset file',
    properties: ['openFile'],
    filters: [
      { name: 'CSV', extensions: ['csv'] },
      { name: 'All files', extensions: ['*'] }
    ]
  });
  return filePaths && filePaths.length > 0 ? filePaths[0] : null;
});

app.whenReady().then(() => {
  startPythonProcesses();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  stopPythonProcesses();

  if (process.platform !== 'darwin') {
    app.quit();
  }
});

