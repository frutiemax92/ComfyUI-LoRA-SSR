import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Rows are lora_<n> widgets ({lora, strength}) each followed by a conditioning_<n> label widget
// carrying the CONDITIONING input socket, numbered 1..count so saved inputs match on reload.
const SLIDER_MIN = 0;
const SLIDER_MAX = 2;
const MARGIN = 15;

async function showLoraMenu(event, callback) {
  const loras = await (await api.fetchApi("/models/loras")).json();
  new LiteGraph.ContextMenu(loras, { event, title: "LoRA", className: "dark", scale: Math.max(1, app.canvas.ds.scale), callback });
}

function fitText(ctx, text, width) {
  if (ctx.measureText(text).width <= width) return text;
  while (text.length && ctx.measureText(text + "…").width > width) text = text.slice(0, -1);
  return text + "…";
}

function rowLayout(width, h) {
  const sliderWidth = Math.min(110, (width - 2 * MARGIN) * 0.4);
  return {
    remove: [MARGIN, MARGIN + h],
    name: [MARGIN + h + 4, width - MARGIN - sliderWidth - 4],
    slider: [width - MARGIN - sliderWidth, width - MARGIN],
  };
}

function rowCount(node) {
  return node.widgets.filter((w) => w.name.startsWith("lora_")).length;
}

function addRow(node, value) {
  const n = rowCount(node) + 1;
  const button = node.widgets.find((w) => w.name === "add_lora");
  if (button) node.removeWidget(button);

  node.addCustomWidget({
    name: `lora_${n}`,
    type: "ssr_lora",
    value: value ?? { lora: null, strength: 1.0 },
    draw(ctx, node, width, y, h) {
      const L = rowLayout(width, h);
      ctx.save();
      ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
      ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
      ctx.beginPath();
      ctx.roundRect(MARGIN, y, width - 2 * MARGIN, h, [h * 0.5]);
      ctx.fill();
      ctx.stroke();

      const t = Math.min(1, Math.max(0, (this.value.strength - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN)));
      ctx.fillStyle = "#4a6a8a";
      ctx.beginPath();
      ctx.roundRect(L.slider[0], y + 3, (L.slider[1] - L.slider[0] - 3) * t, h - 6, [(h - 6) * 0.5]);
      ctx.fill();

      ctx.textBaseline = "middle";
      ctx.textAlign = "center";
      ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
      ctx.fillText("✕", (L.remove[0] + L.remove[1]) / 2, y + h / 2);
      ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
      ctx.fillText(this.value.strength.toFixed(2), (L.slider[0] + L.slider[1]) / 2, y + h / 2);
      ctx.textAlign = "left";
      ctx.fillText(fitText(ctx, this.value.lora ?? "Select a LoRA", L.name[1] - L.name[0]), L.name[0], y + h / 2);
      ctx.restore();
    },
    onPointerDown(pointer, node, canvas) {
      const e = pointer.eDown;
      const L = rowLayout(node.size[0], LiteGraph.NODE_WIDGET_HEIGHT);
      const x = e.canvasX - node.pos[0];
      if (x < L.remove[1]) {
        pointer.onClick = () => removeRow(node, this.name);
      } else if (x >= L.slider[0]) {
        const setFromPointer = (ev) => {
          const t = Math.min(1, Math.max(0, (ev.canvasX - node.pos[0] - L.slider[0]) / (L.slider[1] - L.slider[0])));
          this.value = { ...this.value, strength: Math.round((SLIDER_MIN + t * (SLIDER_MAX - SLIDER_MIN)) * 100) / 100 };
          node.setDirtyCanvas(true);
        };
        pointer.onDragStart = () => setFromPointer(e);
        pointer.onDrag = setFromPointer;
        pointer.onDoubleClick = () => canvas.prompt("Strength", this.value.strength, (v) => {
          const strength = parseFloat(v);
          if (!isNaN(strength)) this.value = { ...this.value, strength };
          node.setDirtyCanvas(true);
        }, e);
      } else {
        pointer.onClick = () => showLoraMenu(e, (lora) => {
          this.value = { ...this.value, lora };
          node.setDirtyCanvas(true);
        });
      }
      return true;
    },
  });

  node.addCustomWidget({
    name: `conditioning_${n}`,
    type: "ssr_conditioning",
    value: null,
    serialize: false,
    options: { serialize: false },
    draw(ctx, node, width, y, h) {
      ctx.save();
      ctx.textBaseline = "middle";
      ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
      const linked = node.isInputConnected(node.inputs.findIndex((i) => i.name === this.name));
      ctx.fillText(linked ? "↳ calibration conditioning" : "↳ connect calibration conditioning", MARGIN + 8, y + h / 2);
      ctx.restore();
    },
  });

  if (!node.inputs?.some((i) => i.name === `conditioning_${n}`)) node.addInput(`conditioning_${n}`, "CONDITIONING");
  const input = node.inputs.find((i) => i.name === `conditioning_${n}`);
  input.widget = { name: `conditioning_${n}` };
  input.alwaysVisible = true;

  node.addWidget("button", "add_lora", null, () => addRow(node), { serialize: false }).label = "+ Add LoRA";
  node.widgets.at(-1).serialize = false;
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.setDirtyCanvas(true, true);
}

function removeRow(node, name) {
  const n = parseInt(name.slice("lora_".length));
  const count = rowCount(node);
  node.removeWidget(node.widgets.find((w) => w.name === `lora_${n}`));
  node.removeWidget(node.widgets.find((w) => w.name === `conditioning_${n}`));
  node.removeInput(node.inputs.findIndex((i) => i.name === `conditioning_${n}`));
  for (let k = n + 1; k <= count; k++) {
    node.widgets.find((w) => w.name === `lora_${k}`).name = `lora_${k - 1}`;
    node.widgets.find((w) => w.name === `conditioning_${k}`).name = `conditioning_${k - 1}`;
    const input = node.inputs.find((i) => i.name === `conditioning_${k}`);
    input.name = `conditioning_${k - 1}`;
    input.widget.name = `conditioning_${k - 1}`;
  }
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "ComfyUI-LoRA-SSR.SSRMergeLoRAs",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SSRMergeLoRAs") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      addRow(this);
      addRow(this);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      onConfigure?.apply(this, arguments);
      const size = [...this.size];
      for (const w of this.widgets.filter((w) => w.name.startsWith("lora_") || w.name.startsWith("conditioning_"))) {
        this.removeWidget(w);
      }
      for (const value of info.widgets_values ?? []) {
        if (value?.lora !== undefined) addRow(this, { ...value });
      }
      this.setSize([size[0], Math.max(size[1], this.computeSize()[1])]);
    };
  },
});
