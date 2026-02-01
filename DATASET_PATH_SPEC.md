# Dataset Path Contract — Frontend ↔ Desktop App

This document describes how the **dataset file path** is chosen in the UI and sent to the backend. It is intended for the team building or integrating the **desktop app** (e.g. Electron wrapper) so they can implement the file picker and path handling correctly.

---

## 1. What the frontend sends to the backend

When the user clicks **Train**, the frontend sends a JSON payload to `POST /train`. The dataset is described by an object that includes a **path** field:

```json
{
  "epochs": 50,
  "dataset": {
    "type": "csv",
    "path": "/absolute/path/on/system/to/data.csv",
    "batch_size": 16
  },
  "blocks": [ ... ]
}
```

- **`dataset.path`** is the **absolute filesystem path** of the dataset file on the machine where the **backend** runs (or where the desktop app runs, if the backend reads files from that same machine).
- The frontend sends this string as-is. No transformation or encoding is applied; it is a normal JSON string.

---

## 2. How the frontend gets the path

Two sources, in order of precedence:

1. **File picker (Browse)**  
   - The user clicks **Browse…** in the “Dataset file” section in the sidebar.  
   - The **desktop app** must provide a native file picker that returns the **absolute path** of the selected file.  
   - That path is stored in the frontend and used as `dataset.path` when building the train payload.

2. **Dataset block**  
   - The user can also type (or paste) a path in the **Dataset** block’s “Path” field.  
   - If the user has **not** chosen a file via Browse, the frontend uses this block path as `dataset.path`.

**Priority:** If a path was chosen via the file picker, it **overrides** the path from the Dataset block for that training request. So: **file picker path (if set) → else Dataset block path.**

---

## 3. Contract for the desktop app (file picker)

The frontend expects the desktop host to expose a single API used only for the dataset file picker.

### 3.1 API shape

- **Name:** `openDatasetFilePicker`
- **Where:** Exposed on `window.electronAPI.openDatasetFilePicker` (or equivalent in your host).
- **Behavior:** When called (e.g. when the user clicks “Browse…”), the desktop app should:
  1. Show a **native “open file” dialog** (e.g. Electron’s `dialog.showOpenDialog` with `openFile`).
  2. Restrict to a single file (no multi-select).
  3. Optionally filter by CSV (and/or other allowed extensions) if desired.
  4. Return the **absolute path** of the selected file.

### 3.2 Return value

- **Success (user picked a file):** Return the **absolute path** of that file as a string (e.g. `"/home/user/data/iris.csv"` or `"C:\\Users\\User\\data\\iris.csv"`).
- **Cancel (user closed the dialog without selecting):** Return `null` or an empty string. The frontend will not change the current path and will not show an error.

The frontend does not send the path to any endpoint other than `POST /train` inside the `dataset.path` field.

### 3.3 When the picker is not available

- If `window.electronAPI.openDatasetFilePicker` is not a function (e.g. running in a browser with no desktop host), the frontend shows a message that the file picker is only available in the desktop app and asks the user to enter the path in the Dataset block instead.
- The backend contract is unchanged: it still receives `dataset.path` as a string (from the block in that case).

---

## 4. Example: Electron implementation

The reference desktop app uses:

1. **Preload script** (`preload.js`)  
   - Exposes `openDatasetFilePicker` via `contextBridge` as `window.electronAPI.openDatasetFilePicker`.  
   - Implementation: call `ipcRenderer.invoke('dialog:openDatasetFile')`.

2. **Main process**  
   - Handle `dialog:openDatasetFile` with `dialog.showOpenDialog(..., { properties: ['openFile'], filters: [ { name: 'CSV', extensions: ['csv'] }, ... ] })`.  
   - Return the first selected path string, or `null` if cancelled.

3. **Renderer**  
   - On “Browse…” click: `const path = await window.electronAPI.openDatasetFilePicker();`  
   - If `path` is non-null/non-empty, store it and use it as `dataset.path` when building the train payload.

---

## 5. Summary for backend / training logic

- The **only** path the backend receives for the dataset is in **`dataset.path`** in the `POST /train` JSON body.
- That value is a **string**: the absolute path of the dataset file.
- The backend (or the desktop app’s training logic) should **read the file from that path** on the same machine where the training process runs. No upload or separate file transfer is implied by this contract; the frontend only sends the path string.
