// Preload script: exposes a safe API to the renderer for the desktop app.
// The renderer calls window.electronAPI.openDatasetFilePicker() to get an absolute file path.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  /**
   * Opens a native file picker for choosing the dataset file.
   * @returns {Promise<string|null>} Resolves with the absolute path of the selected file, or null if cancelled.
   */
  openDatasetFilePicker: () => ipcRenderer.invoke('dialog:openDatasetFile')
});
