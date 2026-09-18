/* Tiny DOM helpers. No innerHTML anywhere in the app: every piece of
   API-derived text goes through textContent / createTextNode, so transcript
   markers like <nzr> render literally instead of being parsed as HTML. */

/** Create an element. `props` keys: class, text, title, style, value, checked,
 *  anything else becomes an attribute; keys starting with "on" become
 *  addEventListener("<rest>", fn) — listeners, never inline handlers. */
export function el(tag, props = {}, kids = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = String(value);
    else if (key === "style") node.setAttribute("style", value);
    else if (key === "value") node.value = value;
    else if (key === "checked" || key === "disabled") node[key] = Boolean(value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2), value);
    } else node.setAttribute(key, String(value));
  }
  for (const kid of Array.isArray(kids) ? kids : [kids]) {
    if (kid === null || kid === undefined || kid === false) continue;
    node.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
  }
  return node;
}

/** Remove every child of `node`. */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** A `.card` section with an uppercase heading. */
export function card(title, kids = [], klass = "") {
  const children = [];
  if (title) children.push(el("h2", { text: title }));
  for (const kid of Array.isArray(kids) ? kids : [kids]) if (kid) children.push(kid);
  return el("div", { class: klass ? `card ${klass}` : "card" }, children);
}

/** A `<span class="chip"><b>value</b> label</span>`. */
export function chip(value, label, klass = "") {
  const kids = [el("b", { text: value })];
  if (label) kids.push(document.createTextNode(" " + label));
  return el("span", { class: klass ? `chip ${klass}` : "chip" }, kids);
}

/** Amber warning banner. */
export function banner(text, klass = "") {
  return el("div", { class: klass ? `banner ${klass}` : "banner", text });
}
