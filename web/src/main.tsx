import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/layout.css";
import "./styles/rail.css";
import "./styles/buttons.css";
import "./styles/stage.css";
import "./styles/hitl.css";
import "./styles/views.css";
import "./styles/cards.css";
import "./styles/topology.css";
import "./styles/table.css";
import "./styles/waterfall.css";
import "./styles/dock.css";
import "./styles/dialog.css";
import "./styles/workbench.css";
import "./styles/responsive.css";

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { Workbench } from "./state/workbench";

const container = document.getElementById("root");
if (!container) throw new Error("Ariadne: #root is missing from index.html");

const workbench = new Workbench();
createRoot(container).render(
  <StrictMode>
    <App workbench={workbench} />
  </StrictMode>,
);
void workbench.start();
