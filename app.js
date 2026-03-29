import {
  ImageSegmenter,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ===================== CONFIG =====================
const PROCESS_H = 360; // Lower = faster, 360 is sweet spot
const BG_COLOR = "#1a1a2e";
const BLUR_RADIUS = 4; // px, mask edge smoothing
const MASK_THRESHOLD_HI = 0.6; // above = fully person
const MASK_THRESHOLD_LO = 0.3; // below = fully background

// ===================== STATE =====================
const S = {
  video: null,
  canvas: null,
  ctx: null,
  segmenter: null,

  // Processing canvases (offscreen)
  procCvs: null,
  procCtx: null,
  personCvs: null,
  personCtx: null,
  maskCvs: null,
  maskCtx: null,

  // Dimensions
  pw: 0,
  ph: 0,

  // Camera
  facingMode: "user",
  stream: null,

  // Person transform
  px: 0,
  py: 0,
  pscale: 1,

  // Background
  bgType: "color", // color | image | video
  bgImg: null,
  bgVid: null,

  // Runtime
  running: false,
  maskData: null,

  // Mask auto-detect
  maskIdx: 0,
  maskInvert: false,
  maskChecked: false,

  // FPS
  fpsFrames: 0,
  fpsLast: 0,
};

// ===================== INIT =====================
async function init() {
  S.video = document.getElementById("camera");
  S.canvas = document.getElementById("output");
  S.ctx = S.canvas.getContext("2d");

  S.procCvs = document.createElement("canvas");
  S.procCtx = S.procCvs.getContext("2d");
  S.personCvs = document.createElement("canvas");
  S.personCtx = S.personCvs.getContext("2d");
  S.maskCvs = document.createElement("canvas");
  S.maskCtx = S.maskCvs.getContext("2d");

  resize();
  window.addEventListener("resize", resize);
  if (screen.orientation)
    screen.orientation.addEventListener("change", () => setTimeout(resize, 200));

  try {
    await startCam();
  } catch (e) {
    return showErr("Permite acceso a la camara para continuar.");
  }

  try {
    setText("Descargando modelo IA...");
    await initSegmenter();
  } catch (e) {
    console.error(e);
    return showErr("Error cargando modelo de IA. Recarga la pagina.");
  }

  setupTouch();
  setupUI();

  document.getElementById("loading").classList.add("hidden");
  S.running = true;
  S.fpsLast = performance.now();
  requestAnimationFrame(loop);
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  S.canvas.width = window.innerWidth * dpr;
  S.canvas.height = window.innerHeight * dpr;
  S.canvas.style.width = window.innerWidth + "px";
  S.canvas.style.height = window.innerHeight + "px";
}

function setText(t) {
  document.getElementById("loading-text").textContent = t;
}
function showErr(msg) {
  const el = document.getElementById("loading");
  el.innerHTML = `<p style="padding:24px;text-align:center;font-size:15px">${msg}</p>`;
}

// ===================== CAMERA =====================
async function startCam() {
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());

  try {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: S.facingMode },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
      audio: false,
    });
  } catch {
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
  }

  S.video.srcObject = S.stream;
  await new Promise((r) => {
    S.video.onloadedmetadata = () => S.video.play().then(r);
  });

  // Force widest FOV — set zoom to minimum
  const track = S.stream.getVideoTracks()[0];
  try {
    const caps = track.getCapabilities ? track.getCapabilities() : {};
    if (caps.zoom) {
      await track.applyConstraints({
        advanced: [{ zoom: caps.zoom.min }],
      });
    }
    if (caps.resizeMode && caps.resizeMode.includes("none")) {
      await track.applyConstraints({
        advanced: [{ resizeMode: "none" }],
      });
    }
  } catch {}

  // Set processing canvas dimensions
  const vw = S.video.videoWidth;
  const vh = S.video.videoHeight;
  const aspect = vw / vh;
  S.ph = PROCESS_H;
  S.pw = Math.round(PROCESS_H * aspect);

  for (const c of [S.procCvs, S.personCvs, S.maskCvs]) {
    c.width = S.pw;
    c.height = S.ph;
  }

  S.maskData = new ImageData(S.pw, S.ph);
  S.maskChecked = false; // Re-detect mask orientation on new camera
}

// ===================== SEGMENTER =====================
async function initSegmenter() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  const opts = {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    outputCategoryMask: false,
    outputConfidenceMasks: true,
  };

  try {
    S.segmenter = await ImageSegmenter.createFromOptions(vision, opts);
  } catch {
    // GPU not available, fallback to CPU
    console.warn("GPU delegate failed, falling back to CPU");
    opts.baseOptions.delegate = "CPU";
    S.segmenter = await ImageSegmenter.createFromOptions(vision, opts);
  }
}

// ===================== RENDER LOOP =====================
function loop(ts) {
  if (!S.running) return;
  process(ts);
  updateFps(ts);
  requestAnimationFrame(loop);
}

function process(ts) {
  if (S.video.readyState < 2 || !S.segmenter) return;

  const { pw, ph, procCtx, procCvs, video } = S;

  // Draw video to processing canvas (mirror front cam)
  procCtx.save();
  if (S.facingMode === "user") {
    procCtx.translate(pw, 0);
    procCtx.scale(-1, 1);
  }
  procCtx.drawImage(video, 0, 0, pw, ph);
  procCtx.restore();

  // Segment
  let result;
  try {
    result = S.segmenter.segmentForVideo(procCvs, ts);
  } catch {
    return;
  }

  if (result.confidenceMasks && result.confidenceMasks.length > 0) {
    // Auto-detect which mask is person on first frame
    if (!S.maskChecked) {
      S.maskChecked = true;
      const masks = result.confidenceMasks;
      // Check center pixel of each mask to find person
      const centerIdx = Math.floor((pw * ph) / 2) + Math.floor(pw / 2);
      if (masks.length === 1) {
        const d = masks[0].getAsFloat32Array();
        // If center is low, person might be inverted
        S.maskInvert = d[centerIdx] < 0.3;
        S.maskIdx = 0;
      } else {
        // Pick mask where center pixel is highest (person is usually in center)
        let bestIdx = 0, bestVal = -1;
        for (let mi = 0; mi < masks.length; mi++) {
          const d = masks[mi].getAsFloat32Array();
          if (d[centerIdx] > bestVal) { bestVal = d[centerIdx]; bestIdx = mi; }
        }
        S.maskIdx = bestIdx;
        S.maskInvert = bestVal < 0.3;
      }
      console.log(`Mask: idx=${S.maskIdx}, invert=${S.maskInvert}, masks=${masks.length}`);
    }

    const cmask = result.confidenceMasks[S.maskIdx];
    const data = cmask.getAsFloat32Array();
    buildPerson(data, pw, ph);
    for (const m of result.confidenceMasks) m.close();
  }

  composite();
}

function buildPerson(mask, w, h) {
  const { personCtx, procCvs, maskCtx, maskCvs, maskData } = S;
  const d = maskData.data;

  // Build alpha mask from confidence values
  for (let i = 0; i < mask.length; i++) {
    let a = S.maskInvert ? 1 - mask[i] : mask[i];
    // Soft threshold — smooth transition
    if (a > MASK_THRESHOLD_HI) a = 1;
    else if (a < MASK_THRESHOLD_LO) a = 0;
    else a = (a - MASK_THRESHOLD_LO) / (MASK_THRESHOLD_HI - MASK_THRESHOLD_LO);

    const j = i << 2;
    d[j] = 255;
    d[j | 1] = 255;
    d[j | 2] = 255;
    d[j | 3] = (a * 255 + 0.5) | 0;
  }
  maskCtx.putImageData(maskData, 0, 0);

  // Person = video composited with blurred mask
  personCtx.clearRect(0, 0, w, h);
  personCtx.drawImage(procCvs, 0, 0);
  personCtx.globalCompositeOperation = "destination-in";
  try {
    personCtx.filter = `blur(${BLUR_RADIUS}px)`;
  } catch {}
  personCtx.drawImage(maskCvs, 0, 0);
  personCtx.filter = "none";
  personCtx.globalCompositeOperation = "source-over";
}

function composite() {
  const { canvas, ctx, personCvs, px, py, pscale } = S;
  const cw = canvas.width;
  const ch = canvas.height;

  ctx.clearRect(0, 0, cw, ch);

  // Draw background
  drawBg(ctx, cw, ch);

  // Draw person (cover fit + user transform)
  if (personCvs.width > 0 && personCvs.height > 0) {
    const pw = personCvs.width;
    const ph = personCvs.height;
    const cover = Math.max(cw / pw, ch / ph);
    const dw = pw * cover * pscale;
    const dh = ph * cover * pscale;
    const dx = (cw - dw) / 2 + px;
    const dy = (ch - dh) / 2 + py;

    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(personCvs, dx, dy, dw, dh);
  }
}

function drawBg(ctx, w, h) {
  if (S.bgType === "image" && S.bgImg) {
    drawCover(ctx, S.bgImg, w, h);
  } else if (S.bgType === "video" && S.bgVid && S.bgVid.readyState >= 2) {
    drawCover(ctx, S.bgVid, w, h);
  } else {
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, w, h);
  }
}

function drawCover(ctx, src, cw, ch) {
  const sw = src.videoWidth || src.naturalWidth || src.width;
  const sh = src.videoHeight || src.naturalHeight || src.height;
  if (!sw || !sh) return;
  const s = Math.max(cw / sw, ch / sh);
  ctx.drawImage(src, (cw - sw * s) / 2, (ch - sh * s) / 2, sw * s, sh * s);
}

// ===================== FPS =====================
function updateFps(ts) {
  S.fpsFrames++;
  if (ts - S.fpsLast >= 1000) {
    const fps = Math.round((S.fpsFrames * 1000) / (ts - S.fpsLast));
    document.getElementById("fps-counter").textContent = fps + " fps";
    S.fpsFrames = 0;
    S.fpsLast = ts;
  }
}

// ===================== TOUCH =====================
function setupTouch() {
  const c = S.canvas;
  const touches = {};
  let pinchD0 = 0,
    pinchS0 = 1;
  let dragX0 = 0,
    dragY0 = 0,
    spx0 = 0,
    spy0 = 0;

  c.addEventListener(
    "touchstart",
    (e) => {
      e.preventDefault();
      for (const t of e.changedTouches)
        touches[t.identifier] = { x: t.clientX, y: t.clientY };

      const ids = Object.keys(touches);
      if (ids.length >= 2) {
        const a = touches[ids[0]],
          b = touches[ids[1]];
        pinchD0 = Math.hypot(a.x - b.x, a.y - b.y) || 1;
        pinchS0 = S.pscale;
      } else if (ids.length === 1) {
        const a = touches[ids[0]];
        dragX0 = a.x;
        dragY0 = a.y;
        spx0 = S.px;
        spy0 = S.py;
      }
    },
    { passive: false }
  );

  c.addEventListener(
    "touchmove",
    (e) => {
      e.preventDefault();
      for (const t of e.changedTouches)
        if (touches[t.identifier])
          touches[t.identifier] = { x: t.clientX, y: t.clientY };

      const ids = Object.keys(touches);
      const dpr = window.devicePixelRatio || 1;

      if (ids.length >= 2) {
        // Pinch to resize
        const a = touches[ids[0]],
          b = touches[ids[1]];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        S.pscale = Math.max(0.1, Math.min(6, pinchS0 * (dist / pinchD0)));
      } else if (ids.length === 1) {
        // Drag
        const a = touches[ids[0]];
        S.px = spx0 + (a.x - dragX0) * dpr;
        S.py = spy0 + (a.y - dragY0) * dpr;
      }
    },
    { passive: false }
  );

  const onEnd = (e) => {
    for (const t of e.changedTouches) delete touches[t.identifier];
    const ids = Object.keys(touches);
    if (ids.length === 1) {
      // Transition pinch → drag
      const a = touches[ids[0]];
      dragX0 = a.x;
      dragY0 = a.y;
      spx0 = S.px;
      spy0 = S.py;
    }
  };
  c.addEventListener("touchend", onEnd, { passive: false });
  c.addEventListener("touchcancel", onEnd, { passive: false });
}

// ===================== UI =====================
function setupUI() {
  // Flip camera
  document.getElementById("btn-flip").addEventListener("click", async () => {
    S.facingMode = S.facingMode === "user" ? "environment" : "user";
    await startCam();
  });

  // Background image
  const fileImg = document.getElementById("file-image");
  document
    .getElementById("btn-bg-image")
    .addEventListener("click", () => fileImg.click());
  fileImg.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const img = new Image();
    img.onload = () => {
      S.bgImg = img;
      S.bgType = "image";
    };
    img.src = URL.createObjectURL(f);
    e.target.value = "";
  });

  // Background video
  const fileVid = document.getElementById("file-video");
  document
    .getElementById("btn-bg-video")
    .addEventListener("click", () => fileVid.click());
  fileVid.addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const v = document.getElementById("bg-video");
    v.src = URL.createObjectURL(f);
    v.play().catch(() => {});
    S.bgVid = v;
    S.bgType = "video";
    e.target.value = "";
  });

  // Reset position/scale
  document.getElementById("btn-reset").addEventListener("click", () => {
    S.px = 0;
    S.py = 0;
    S.pscale = 1;
  });

  // Clear background
  document.getElementById("btn-clear-bg").addEventListener("click", () => {
    S.bgType = "color";
    S.bgImg = null;
    const v = document.getElementById("bg-video");
    v.pause();
    v.removeAttribute("src");
    v.load();
    S.bgVid = null;
  });
}

// ===================== START =====================
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
