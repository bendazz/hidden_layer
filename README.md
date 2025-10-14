# Feedforward Network Visualizer (3-4-1)

A tiny, framework-free web app that visualizes a feedforward neural network for binary classification:

- Inputs: X0 (bias, all ones), X1, X2
- Hidden layer: 4 ReLU units
- Output: 1 Sigmoid unit
- Displays: data table, network diagram with weights, and average log loss

All files are in the repository root: `index.html`, `styles.css`, `app.js`.

## How to run

You can open `index.html` directly in your browser, or serve the folder locally.

Option 1: open the file directly (double-click or drag-drop into a browser).

Option 2: run a local server and navigate to http://localhost:8000

```bash
python3 -m http.server 8000
```

## Customize

- Edit the dataset and initial weights in `app.js`.
- The network diagram uses SVG; you can tweak layout/labels in `app.js` and styles in `styles.css`.
