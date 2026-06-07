import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "handsontable/styles/handsontable.css";
import "handsontable/styles/ht-theme-main.css";
import "./monacoWorkers";
import "./styles.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
