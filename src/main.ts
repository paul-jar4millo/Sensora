import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs-core";

// Referencias
const video = document.getElementById("webcam") as HTMLVideoElement;
const canvas = document.getElementById("overlay") as HTMLCanvasElement;
const statusBadge = document.getElementById("status-badge") as HTMLDivElement;
const FPS_LIMIT = 10;
const HISTORY_SIZE = 6;
const MIN_CONFIDENCE = 4;
const SPEECH_COOLDOWN_MS = 5000;

let frameHistory: string[][] = [];
let lastSpokenTime: Record<string, number> = {};
let previousDetections: cocoSsd.DetectedObject[] = [];

// Canvas oculto para análisis de proximidad por diferencia de frames
const proximityCanvas = document.createElement("canvas");
const proximityCtx = proximityCanvas.getContext("2d", { willReadFrequently: true })!;
let previousFrameData: ImageData | null = null;


// Traducciones de objetos
const TRANSLATIONS: Record<string, string> = {
  "person": "persona",
  "car": "auto",
  "truck": "camión",
  "bus": "autobús",
  "motorcycle": "motocicleta",
  "bicycle": "bicicleta",
  "stop sign": "señal de alto",
  "fire hydrant": "hidrante",
  "parking meter": "parquímetro",
  "bench": "banco",
  "chair": "silla",
  "couch": "sofá",
  "bed": "cama",
  "dining table": "mesa",
  "tv": "televisor",
  "laptop": "laptop",
  "cell phone": "celular",
  "book": "libro",
  "clock": "reloj",
  "bottle": "botella",
  "cup": "taza",
  "potted plant": "planta",
  "stairs": "escaleras",
  "pothole": "bache",
  "door": "puerta",
  "wall": "pared",
  "pole": "poste",
  "bird": "pájaro",
  "cat": "gato",
  "dog": "perro",
  "horse": "caballo",
  "sheep": "oveja",
  "cow": "vaca",
  "elephant": "elefante",
  "bear": "oso",
  "zebra": "cebra",
  "giraffe": "jirafa",
  "backpack": "mochila",
  "umbrella": "paraguas",
  "handbag": "bolso",
  "tie": "corbata",
  "suitcase": "maleta",
  "unknown": "obstáculo"
};

function translateClass(className: string): string {
  return TRANSLATIONS[className] || className;
}

// Jerarquía de riesgo
const RISK_LEVELS = {
  CRITICAL: 3,
  OBSTACLE: 2,
  INFORMATIVE: 1,
  UNKNOWN: 0
};

const RISK_MAP: Record<string, number> = {
  // Críticos (Acción inmediata)
  "stairs": RISK_LEVELS.CRITICAL,
  "car": RISK_LEVELS.CRITICAL,
  "truck": RISK_LEVELS.CRITICAL,
  "bus": RISK_LEVELS.CRITICAL,
  "motorcycle": RISK_LEVELS.CRITICAL,
  "pothole": RISK_LEVELS.CRITICAL,

  // Obstáculos (Navegación)
  "person": RISK_LEVELS.OBSTACLE,
  "stop sign": RISK_LEVELS.OBSTACLE,
  "fire hydrant": RISK_LEVELS.OBSTACLE,
  "parking meter": RISK_LEVELS.OBSTACLE,
  "bench": RISK_LEVELS.OBSTACLE,
  "door": RISK_LEVELS.OBSTACLE,
  "wall": RISK_LEVELS.OBSTACLE,
  "pole": RISK_LEVELS.OBSTACLE,
  "unknown": RISK_LEVELS.OBSTACLE,

  // Informativos (Contexto)
  "chair": RISK_LEVELS.INFORMATIVE,
  "dining table": RISK_LEVELS.INFORMATIVE,
  "couch": RISK_LEVELS.INFORMATIVE,
  "bed": RISK_LEVELS.INFORMATIVE,
  "tv": RISK_LEVELS.INFORMATIVE,
  "laptop": RISK_LEVELS.INFORMATIVE
};

function getRiskLevel(className: string): number {
  return RISK_MAP[className] || RISK_LEVELS.UNKNOWN;
}

function getCenter(bbox: [number, number, number, number]) {
  return { x: bbox[0] + bbox[2] / 2, y: bbox[1] + bbox[3] / 2 };
}

// Zona central
function isCentralZone(bbox: [number, number, number, number], frameWidth: number, frameHeight: number): boolean {
  const center = getCenter(bbox);
  const horizonY = frameHeight * 0.4;
  if (center.y < horizonY) return false;
  const progress = (center.y - horizonY) / (frameHeight - horizonY);
  const leftX = frameWidth * (0.4 - 0.25 * progress);
  const rightX = frameWidth * (0.6 + 0.25 * progress);
  return center.x >= leftX && center.x <= rightX;
}

// Detecta si un objeto se acerca al centro
function isMovingTowardsCenter(current: cocoSsd.DetectedObject, prevs: cocoSsd.DetectedObject[], frameWidth: number, frameHeight: number): boolean {
  const currentCenter = getCenter(current.bbox);
  const frameCenter = { x: frameWidth / 2, y: frameHeight / 2 };

  const sameClassPrevs = prevs.filter(p => p.class === current.class);
  if (sameClassPrevs.length === 0) return false;

  let closestPrev = sameClassPrevs[0];
  let minDistance = Infinity;
  for (const prev of sameClassPrevs) {
    const prevCenter = getCenter(prev.bbox);
    const dist = Math.hypot(prevCenter.x - currentCenter.x, prevCenter.y - currentCenter.y);
    if (dist < minDistance) {
      minDistance = dist;
      closestPrev = prev;
    }
  }

  const prevCenter = getCenter(closestPrev.bbox);
  const prevDistToCenter = Math.hypot(prevCenter.x - frameCenter.x, prevCenter.y - frameCenter.y);
  const currentDistToCenter = Math.hypot(currentCenter.x - frameCenter.x, currentCenter.y - frameCenter.y);

  return currentDistToCenter < prevDistToCenter - 2;
}

// Configuración de la cámara
async function setupCamera(): Promise<void> {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    video.srcObject = stream;
  } catch (error) {
    console.error("Error accediendo a la cámara:", error);
  }
}

// Inicialización
async function init() {
  await tf.ready();
  console.log("Backend activo:", tf.getBackend());

  const model = await cocoSsd.load();
  console.log("Modelo COCO-SSD cargado correctamente");

  statusBadge.innerText = "Modelo Listo";
  statusBadge.classList.add("ready");

  predictLoop(model);
}

// Detección de obstáculos desconocidos por diferencia de frames (sin dependencias externas)
function analyzeProximity(): cocoSsd.DetectedObject[] {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return [];

  // Escalar a un tamaño pequeño para mayor rendimiento
  const sampleW = 64;
  const sampleH = 48;
  proximityCanvas.width = sampleW;
  proximityCanvas.height = sampleH;

  proximityCtx.drawImage(video, 0, 0, sampleW, sampleH);
  const currentFrame = proximityCtx.getImageData(0, 0, sampleW, sampleH);

  if (!previousFrameData) {
    previousFrameData = currentFrame;
    return [];
  }

  // Zona central (40%–60% x, 55%–85% y) del frame muestreado
  const x0 = Math.floor(sampleW * 0.35);
  const x1 = Math.floor(sampleW * 0.65);
  const y0 = Math.floor(sampleH * 0.55);
  const y1 = Math.floor(sampleH * 0.85);

  let changedPixels = 0;
  let totalPixels = 0;

  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * sampleW + x) * 4;
      const dr = Math.abs(currentFrame.data[i]     - previousFrameData.data[i]);
      const dg = Math.abs(currentFrame.data[i + 1] - previousFrameData.data[i + 1]);
      const db = Math.abs(currentFrame.data[i + 2] - previousFrameData.data[i + 2]);
      const diff = (dr + dg + db) / 3;
      if (diff > 25) changedPixels++; // umbral de cambio significativo
      totalPixels++;
    }
  }

  previousFrameData = currentFrame;

  const changeRatio = changedPixels / totalPixels;

  // Si más del 30% de la zona central está cambiando (masa en movimiento acercandose)
  if (changeRatio > 0.30) {
    return [{
      bbox: [width * 0.25, height * 0.45, width * 0.5, height * 0.45],
      class: "unknown",
      score: changeRatio
    }];
  }

  return [];
}

function processDetections(predictions: cocoSsd.DetectedObject[]): cocoSsd.DetectedObject[] {
  const currentClasses = Array.from(new Set(predictions.map(p => p.class)));

  frameHistory.push(currentClasses);
  if (frameHistory.length > HISTORY_SIZE) {
    frameHistory.shift();
  }

  const classCounts: Record<string, number> = {};
  for (const frame of frameHistory) {
    for (const className of frame) {
      classCounts[className] = (classCounts[className] || 0) + 1;
    }
  }

  const confidentClasses = Object.keys(classCounts).filter(
    className => classCounts[className] >= MIN_CONFIDENCE
  );
  const confidentPredictions = predictions.filter(p => confidentClasses.includes(p.class));

  const frameWidth = video.videoWidth || 640;
  const frameHeight = video.videoHeight || 480;

  const predictionsWithContext = confidentPredictions.map(p => {
    const isCentral = isCentralZone(p.bbox, frameWidth, frameHeight);
    const isMoving = isMovingTowardsCenter(p, previousDetections, frameWidth, frameHeight);
    return {
      prediction: p,
      isCentral,
      isMoving,
      isValid: isCentral || isMoving // Mantiene periféricos solo si se acercan al centro
    };
  }).filter(item => item.isValid);

  if (predictionsWithContext.length > 0) {
    const predictionsWithScores = predictionsWithContext.map(item => {
      let score = getRiskLevel(item.prediction.class) * 10;
      if (item.isCentral) score += 8;
      if (item.isMoving) score += 5;
      return { ...item, score };
    });

    // Encontrar el score máximo en el frame actual
    const maxScore = Math.max(...predictionsWithScores.map(item => item.score));

    // Filtrar para quedarse solo con los objetos del mayor score
    const priorityItems = predictionsWithScores.filter(item => item.score === maxScore);

    // Notificar por voz
    const now = Date.now();
    for (const item of priorityItems) {
      const className = item.prediction.class;
      if (!lastSpokenTime[className] || now - lastSpokenTime[className] > SPEECH_COOLDOWN_MS) {
        notifyVoice(item.prediction, item.isMoving, frameWidth);
        lastSpokenTime[className] = now;
      }
    }
  }

  previousDetections = confidentPredictions;

  return predictionsWithContext.map(item => item.prediction);
}

function notifyVoice(prediction: cocoSsd.DetectedObject, isMoving: boolean, frameWidth: number) {
  const translatedName = translateClass(prediction.class);
  const center = getCenter(prediction.bbox);

  let position = "en frente";
  if (center.x < frameWidth * 0.33) {
    position = "a la izquierda";
  } else if (center.x > frameWidth * 0.66) {
    position = "a la derecha";
  }

  let textToSpeak = `${translatedName} ${position}`;
  if (isMoving) {
    textToSpeak += ", acercándose";
  }

  const utterance = new SpeechSynthesisUtterance(textToSpeak);
  utterance.lang = "es-ES";
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

function drawPredictions(predictions: cocoSsd.DetectedObject[]) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height); // Limpiar el canvas del frame anterior

  // Dibujar Camino Predictivo 
  const bottomY = canvas.height;
  const horizonY = canvas.height * 0.4;

  const leftBottomX = canvas.width * 0.15;
  const rightBottomX = canvas.width * 0.85;

  const leftHorizonX = canvas.width * 0.4;
  const rightHorizonX = canvas.width * 0.6;

  // Relleno sutil del camino
  ctx.beginPath();
  ctx.moveTo(leftBottomX, bottomY);
  ctx.lineTo(leftHorizonX, horizonY);
  ctx.lineTo(rightHorizonX, horizonY);
  ctx.lineTo(rightBottomX, bottomY);
  ctx.closePath();
  ctx.fillStyle = "rgba(0, 255, 255, 0.05)"; // Cyan muy sutil
  ctx.fill();

  // Líneas laterales predictivas
  ctx.strokeStyle = "rgba(0, 255, 255, 0.6)"; // Cyan brillante
  ctx.lineWidth = 3;

  ctx.beginPath();
  ctx.moveTo(leftBottomX, bottomY);
  ctx.lineTo(leftHorizonX, horizonY);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(rightBottomX, bottomY);
  ctx.lineTo(rightHorizonX, horizonY);
  ctx.stroke();

  // Dibujar objeto detectado
  predictions.forEach(prediction => {
    const risk = getRiskLevel(prediction.class);

    let color = "#00ff22ff"; // Informativo / Desconocido (Verde)
    if (risk === RISK_LEVELS.CRITICAL) {
      color = "#ff0000ff"; // Rojo para Críticos
    } else if (risk === RISK_LEVELS.OBSTACLE) {
      color = "#ffaa00ff"; // Naranja para Obstáculos
    }

    const [x, y, width, height] = prediction.bbox;
    const translatedName = translateClass(prediction.class);
    const text = `${translatedName} (${Math.round(prediction.score * 100)}%)`;

    // Dibujar el cuadro
    ctx.strokeStyle = color;
    ctx.lineWidth = risk === RISK_LEVELS.CRITICAL ? 6 : 4;
    ctx.strokeRect(x, y, width, height);

    // Dibujar un fondo para el texto
    ctx.font = "bold 16px Arial";
    const textWidth = ctx.measureText(text).width;
    const textHeight = 24;

    // Posición Y del texto
    const textY = y > textHeight ? y - textHeight : y;
    ctx.fillStyle = color;
    ctx.fillRect(x, textY, textWidth + 10, textHeight);

    // Texto
    ctx.fillStyle = "#000000";
    ctx.fillText(text, x + 5, textY + 17);
  });
}

async function predictLoop(model: cocoSsd.ObjectDetection) {
  const rawPredictions = await model.detect(video);
  const proximityPredictions = analyzeProximity();

  const allPredictions = [...rawPredictions, ...proximityPredictions];
  const confidentPredictions = processDetections(allPredictions);

  drawPredictions(confidentPredictions);

  setTimeout(() => {
    requestAnimationFrame(() => predictLoop(model));
  }, 1000 / FPS_LIMIT);
}

const startBtn = document.getElementById("start-btn") as HTMLButtonElement;
const startScreen = document.getElementById("start-screen") as HTMLDivElement;

startBtn.addEventListener("click", () => {
  const unlockUtterance = new SpeechSynthesisUtterance("Iniciando Sensora");
  unlockUtterance.lang = "es-ES";
  unlockUtterance.volume = 1.0;
  window.speechSynthesis.speak(unlockUtterance);

  startScreen.style.display = "none";
  setupCamera().then(init);
});
