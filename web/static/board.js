/**
 * Xiangqi play UI — pieces on line intersections; Red bottom (ranks 7–9).
 */

const PIECE_GLYPH = {
  K: "帥",
  A: "仕",
  B: "相",
  N: "傌",
  R: "俥",
  C: "炮",
  P: "兵",
  k: "將",
  a: "士",
  b: "象",
  n: "馬",
  r: "車",
  c: "砲",
  p: "卒",
};

const FILES = "abcdefghi";

/** Pause before first greedy move on a new game. */
const PAUSE_BEFORE_FIRST_ALLY_MS = 900;
/** Pause after Black moves, before next greedy Red move. */
const PAUSE_BEFORE_ALLY_AFTER_ENGINE_MS = 750;
/** Pause after Red move before calling engine. */
const PAUSE_AFTER_ALLY_MS = 600;
/** Pause after engine move before next loop iteration. */
const PAUSE_AFTER_ENGINE_MS = 500;

const ENGINE_FETCH_TIMEOUT_MS = 15 * 60 * 1000;
const API_RETRY_DELAY_MS = 400;
const MAX_API_RETRIES = 3;

let state = null;
let selectedFrom = null;
let legalTargets = [];
let busy = false;
/** Bumped on each new game to cancel in-flight greedy loops. */
let playGeneration = 0;
let lastError = null;

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const overlayEl = document.getElementById("overlay");
const overlayTextEl = document.getElementById("overlay-text");
const thinkingEl = document.getElementById("thinking");
const errorBarEl = document.getElementById("error-bar");
const errorTextEl = document.getElementById("error-text");
const allyModeInputs = document.querySelectorAll('input[name="ally-mode"]');

function sq(row, col) {
  return `${FILES[col]}${row}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isActive(gen) {
  return gen === playGeneration;
}

function allyMode() {
  return state?.allyMode || "human";
}

function humanSide() {
  return state?.humanSide || "red";
}

function engineKind() {
  return state?.engineKind || "llm";
}

function isHumanPiece(ch, isRed) {
  if (ch === ".") return false;
  return humanSide() === "red" ? isRed : !isRed;
}

function selectedHumanSide() {
  const el = document.querySelector('input[name="human-side"]:checked');
  return el ? el.value : "red";
}

function applyEngineKindUI() {
  const kind = engineKind();
  const sideCtl = document.getElementById("side-controls");
  const allyCtl = document.getElementById("ally-controls");
  if (sideCtl) sideCtl.classList.toggle("hidden", kind !== "muzero");
  if (allyCtl) allyCtl.classList.toggle("hidden", kind === "muzero");
  const sub = document.getElementById("subtitle");
  if (sub) {
    sub.textContent =
      kind === "muzero"
        ? "Human vs MuZero (canonical net, full-strength search)"
        : "Red (ally) vs Black (ep_40 engine)";
  }
}

function formatApiError(path, status, detail, rawText) {
  if (detail && typeof detail === "object" && detail.message) {
    const parts = [
      `${path} → ${status}: ${detail.message}`,
      detail.turn != null ? `turn=${detail.turn}` : null,
      detail.sideToMove != null ? `sideToMove=${detail.sideToMove}` : null,
      detail.lastAllyMove ? `lastRed=${detail.lastAllyMove}` : null,
      detail.lastEngineMove ? `lastBlack=${detail.lastEngineMove}` : null,
    ].filter(Boolean);
    return parts.join(" · ");
  }
  if (typeof detail === "string" && detail) {
    return `${path} → ${status}: ${detail}`;
  }
  return `${path} → ${status}: ${rawText || "request failed"}`;
}

function showError(msg) {
  lastError = msg;
  if (errorBarEl && errorTextEl) {
    errorTextEl.textContent = msg;
    errorBarEl.classList.remove("hidden");
  }
  statusEl.textContent = msg;
}

function clearError() {
  lastError = null;
  if (errorBarEl) {
    errorBarEl.classList.add("hidden");
  }
}

async function syncState() {
  state = await fetchJson("/api/state", { method: "GET" });
  return state;
}

async function fetchJson(path, options = {}) {
  const { timeoutMs = 120000, ...fetchOpts } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      ...fetchOpts,
    });
    const rawText = await res.text();
    let body = {};
    try {
      body = rawText ? JSON.parse(rawText) : {};
    } catch {
      body = { detail: rawText };
    }
    if (!res.ok) {
      const err = new Error(formatApiError(path, res.status, body.detail, rawText));
      err.status = res.status;
      err.path = path;
      err.detail = body.detail;
      throw err;
    }
    return body;
  } catch (e) {
    if (e.name === "AbortError") {
      const err = new Error(`${path} timed out after ${Math.round(timeoutMs / 1000)}s`);
      err.status = 408;
      err.path = path;
      throw err;
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

/** POST with resync + retry on turn mismatch (400). */
async function postWithRetry(path, body, { gen, timeoutMs = 120000 } = {}) {
  let lastErr = null;
  for (let attempt = 0; attempt < MAX_API_RETRIES; attempt++) {
    if (!isActive(gen)) {
      throw new Error("Cancelled (new game started)");
    }
    try {
      return await fetchJson(path, {
        method: "POST",
        body: body != null ? JSON.stringify(body) : undefined,
        timeoutMs,
      });
    } catch (e) {
      lastErr = e;
      const canRetry =
        attempt < MAX_API_RETRIES - 1 &&
        (e.status === 400 || e.status === 408);
      if (!canRetry) {
        throw e;
      }
      await syncState();
      if (!isActive(gen)) {
        throw e;
      }
      const msg = `[retry ${attempt + 2}/${MAX_API_RETRIES}] ${e.message}`;
      console.warn(msg);
      showError(msg);
      await sleep(API_RETRY_DELAY_MS);
    }
  }
  throw lastErr;
}

function winnerLabel(w) {
  if (w === "red") return "Red wins";
  if (w === "black") return "Black wins";
  if (w === "draw") return "Draw";
  return "";
}

function parseMove(mv) {
  if (!mv || mv.length < 4) return { from: null, to: null };
  return { from: mv.slice(0, 2), to: mv.slice(2, 4) };
}

function updateStatus() {
  if (!state) return;
  if (lastError && !busy) {
    return;
  }
  if (state.gameOver) {
    statusEl.textContent = winnerLabel(state.winner) || "Game over";
    return;
  }
  if (busy || state.engineThinking) {
    return;
  }
  if (state.turn === "human") {
    statusEl.textContent =
      humanSide() === "red"
        ? "Your turn — click a Red piece"
        : "Your turn — click a Black piece";
  } else if (state.turn === "greedy") {
    statusEl.textContent = "Greedy ally playing…";
  } else if (state.turn === "engine") {
    statusEl.textContent = "Waiting for engine…";
  } else {
    statusEl.textContent = `Paused (turn=${state.turn || "?"})`;
  }
}

function drawGridSvg() {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "board-lines");
  svg.setAttribute("viewBox", "-0.04 -0.04 8.08 9.08");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  const ns = "http://www.w3.org/2000/svg";

  function addLine(x1, y1, x2, y2) {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", String(x1));
    line.setAttribute("y1", String(y1));
    line.setAttribute("x2", String(x2));
    line.setAttribute("y2", String(y2));
    svg.appendChild(line);
  }

  for (let y = 0; y <= 9; y++) {
    addLine(0, y, 8, y);
  }
  for (let x = 0; x <= 8; x++) {
    if (x === 0 || x === 8) {
      addLine(x, 0, x, 9);
    } else {
      addLine(x, 0, x, 4);
      addLine(x, 5, x, 9);
    }
  }
  addLine(3, 0, 5, 2);
  addLine(5, 0, 3, 2);
  addLine(3, 7, 5, 9);
  addLine(5, 7, 3, 9);

  return svg;
}

function positionIntersection(el, row, col) {
  el.style.left = `calc(var(--piece-size) / 2 + ${col} * var(--cell))`;
  el.style.top = `calc(var(--piece-size) / 2 + ${row} * var(--cell))`;
}

function renderBoard() {
  boardEl.innerHTML = "";
  if (!state?.board) return;

  boardEl.appendChild(drawGridSvg());

  const river = document.createElement("div");
  river.className = "river-overlay";
  river.textContent = "楚河          汉界";
  boardEl.appendChild(river);

  const layer = document.createElement("div");
  layer.className = "intersections";

  const allyHl = parseMove(state.lastAllyMove);
  const engHl = parseMove(state.lastEngineMove);
  const humanTurn =
    state.turn === "human" && !state.gameOver && !busy && !state.engineThinking;

  boardEl.classList.toggle("disabled", !humanTurn || busy);

  for (let row = 0; row < 10; row++) {
    for (let col = 0; col < 9; col++) {
      const coord = sq(row, col);
      const ch = state.board[row][col];
      const isRed = ch !== "." && ch === ch.toUpperCase();

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "intersection";
      btn.dataset.row = String(row);
      btn.dataset.col = String(col);
      positionIntersection(btn, row, col);

      if (humanTurn && isHumanPiece(ch, isRed)) btn.classList.add("selectable");
      if (selectedFrom === coord) btn.classList.add("selected");
      const isLegalTarget = selectedFrom && legalTargets.includes(coord);
      if (isLegalTarget) {
        btn.classList.add("legal-target");
      }
      if (allyHl.from === coord) btn.classList.add("ally-from");
      if (allyHl.to === coord) btn.classList.add("ally-to");
      if (engHl.from === coord) btn.classList.add("engine-from");
      if (engHl.to === coord) btn.classList.add("engine-to");

      if (ch !== ".") {
        const span = document.createElement("span");
        span.className = `piece ${isRed ? "red" : "black"}`;
        span.textContent = PIECE_GLYPH[ch] || ch;
        btn.appendChild(span);
      }
      if (isLegalTarget) {
        const marker = document.createElement("span");
        marker.className = "legal-marker";
        marker.setAttribute("aria-hidden", "true");
        btn.appendChild(marker);
      }

      btn.addEventListener("click", () => onIntersectionClick(row, col, ch, isRed));
      layer.appendChild(btn);
    }
  }

  boardEl.appendChild(layer);
}

function refresh() {
  updateStatus();
  renderBoard();
  if (state?.gameOver) {
    overlayTextEl.textContent = winnerLabel(state.winner) || "Game over";
    overlayEl.classList.remove("hidden");
  } else {
    overlayEl.classList.add("hidden");
  }
}

async function runEngineMove(gen) {
  if (!state || state.gameOver || state.turn !== "engine" || !isActive(gen)) {
    return false;
  }
  const t0 = Date.now();
  statusEl.textContent = "Engine thinking… (0s)";
  const tick = setInterval(() => {
    const sec = Math.floor((Date.now() - t0) / 1000);
    statusEl.textContent = `Engine thinking… (${sec}s)`;
  }, 1000);

  try {
    state = await postWithRetry("/api/engine/move", null, {
      gen,
      timeoutMs: ENGINE_FETCH_TIMEOUT_MS,
    });
    clearError();
    return true;
  } finally {
    clearInterval(tick);
    refresh();
  }
}

async function runGreedyGameLoop(gen, { firstPlies = false } = {}) {
  if (!state || state.gameOver || allyMode() !== "greedy" || !isActive(gen)) {
    return;
  }
  busy = true;
  clearError();
  refresh();

  let needPauseBeforeAlly = firstPlies;

  try {
    while (state && !state.gameOver && isActive(gen)) {
      if (state.turn === "greedy") {
        if (needPauseBeforeAlly) {
          statusEl.textContent = "Greedy ally preparing…";
          refresh();
          await sleep(PAUSE_BEFORE_FIRST_ALLY_MS);
          needPauseBeforeAlly = false;
        } else {
          await sleep(PAUSE_BEFORE_ALLY_AFTER_ENGINE_MS);
        }
        if (!isActive(gen)) break;

        state = await postWithRetry("/api/ally/greedy", null, { gen });
        clearError();
        refresh();
        await sleep(PAUSE_AFTER_ALLY_MS);
      } else if (state.turn === "engine") {
        const ok = await runEngineMove(gen);
        if (!ok || !isActive(gen)) break;
        await sleep(PAUSE_AFTER_ENGINE_MS);
      } else {
        await syncState();
        if (!isActive(gen)) break;
        if (state.turn === "greedy" || state.turn === "engine") {
          continue;
        }
        showError(
          `Game loop stopped: unexpected turn "${state.turn}" (sideToMove=${state.sideToMove})`
        );
        break;
      }
    }
  } catch (e) {
    console.error(e);
    showError(e.message);
    try {
      await syncState();
    } catch {
      /* ignore */
    }
  } finally {
    busy = false;
    refresh();
  }
}

async function resumeFromServer() {
  const gen = playGeneration;
  clearError();
  busy = true;
  refresh();
  try {
    await syncState();
    if (state.gameOver) return;
    if (state.turn === "engine") {
      await runEngineMove(gen);
      if (allyMode() === "greedy" && !state.gameOver) {
        await runGreedyGameLoop(gen, { firstPlies: false });
      }
    } else if (state.turn === "greedy" && allyMode() === "greedy") {
      await runGreedyGameLoop(gen, { firstPlies: false });
    }
  } catch (e) {
    showError(e.message);
  } finally {
    busy = false;
    refresh();
  }
}

async function afterAllyMoveShown(gen) {
  refresh();
  await sleep(PAUSE_AFTER_ALLY_MS);
  if (!state.gameOver && state.turn === "engine" && isActive(gen)) {
    await runEngineMove(gen);
    await sleep(PAUSE_AFTER_ENGINE_MS);
    if (allyMode() === "greedy" && !state.gameOver && state.turn === "greedy") {
      await runGreedyGameLoop(gen, { firstPlies: false });
    }
  }
}

async function onIntersectionClick(row, col, ch, isRed) {
  if (!state || state.gameOver || busy || state.engineThinking) return;
  if (state.turn !== "human") return;

  const coord = sq(row, col);
  const gen = playGeneration;

  if (selectedFrom && legalTargets.includes(coord)) {
    const move = `${selectedFrom}${coord}`;
    selectedFrom = null;
    legalTargets = [];
    busy = true;
    boardEl.classList.add("disabled");
    try {
      state = await postWithRetry("/api/move", { move }, { gen });
      clearError();
      await afterAllyMoveShown(gen);
    } catch (e) {
      showError(e.message);
    } finally {
      busy = false;
      refresh();
    }
    return;
  }

  if (!isHumanPiece(ch, isRed)) {
    selectedFrom = null;
    legalTargets = [];
    renderBoard();
    return;
  }

  selectedFrom = coord;
  try {
    const data = await fetchJson(
      `/api/legal?from=${encodeURIComponent(coord)}`,
      { method: "GET" }
    );
    legalTargets = data.targets || [];
    clearError();
  } catch (e) {
    legalTargets = [];
    showError(e.message);
  }
  renderBoard();
}

function selectedAllyMode() {
  const el = document.querySelector('input[name="ally-mode"]:checked');
  return el ? el.value : "human";
}

async function newGame() {
  playGeneration += 1;
  const gen = playGeneration;
  selectedFrom = null;
  legalTargets = [];
  busy = false;
  clearError();
  overlayEl.classList.add("hidden");

  const mode = selectedAllyMode();
  const side = selectedHumanSide();
  state = await postWithRetry(
    "/api/game/new",
    { allyMode: mode, humanSide: side },
    { gen, timeoutMs: 30000 }
  );
  applyEngineKindUI();
  refresh();

  if (!state.gameOver && state.turn === "engine" && isActive(gen)) {
    await runEngineMove(gen); // human plays Black: model (Red) opens
  }
  if (mode === "greedy" && allyMode() === "greedy" && !state.gameOver && isActive(gen)) {
    await runGreedyGameLoop(gen, { firstPlies: true });
  }
}

document.getElementById("btn-new-game").addEventListener("click", newGame);
document.getElementById("btn-overlay-new").addEventListener("click", newGame);
const btnRetry = document.getElementById("btn-retry");
if (btnRetry) {
  btnRetry.addEventListener("click", resumeFromServer);
}

async function init() {
  try {
    await syncState();
    applyEngineKindUI();
    const mode = state.allyMode || "human";
    allyModeInputs.forEach((inp) => {
      inp.checked = inp.value === mode;
    });
    refresh();
    // Do not auto-start greedy on page load (prevents duplicate loops vs New game).
    if (state.turn === "engine") {
      statusEl.textContent =
        "Engine to move — click Retry turn or New game";
    }
  } catch (e) {
    showError(`Failed to connect: ${e.message}`);
  }
}

init();
