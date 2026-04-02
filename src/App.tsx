import { useEffect, useRef, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { 
  Eraser, 
  Palette, 
  Type, 
  Trash2, 
  Camera, 
  CameraOff, 
  Maximize2, 
  Minimize2,
  MousePointer2,
  Hand,
  Grab,
  CircleStop
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from './lib/utils';

// --- Types ---

type Point = { x: number; y: number };

type DrawingLine = {
  points: Point[];
  color: string;
  thickness: number;
  glow: number;
  id: string;
};

type Gesture = 'POINT' | 'PALM' | 'PINCH' | 'FIST' | 'NONE';

const NEON_COLORS = [
  { name: 'Red', value: '#ff3131' },
  { name: 'Blue', value: '#00d2ff' },
  { name: 'Green', value: '#39ff14' },
  { name: 'Yellow', value: '#fff01f' },
  { name: 'Purple', value: '#bc13fe' },
  { name: 'Pink', value: '#ff00ff' },
  { name: 'White', value: '#ffffff' },
];

export default function App() {
  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number>(null);
  const lastPointRef = useRef<Point | null>(null);
  const activeLineIdRef = useRef<string | null>(null);
  const movingLineIdRef = useRef<string | null>(null);
  const pinchStartPointRef = useRef<Point | null>(null);
  const linesRef = useRef<DrawingLine[]>([]);
  const lastCreatedLineIdRef = useRef<string | null>(null);
  const isRecentLineReadyRef = useRef<boolean>(false);

  // --- State ---
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);
  const [color, setColor] = useState(NEON_COLORS[1].value);
  const [thickness, setThickness] = useState(5);
  const [glowIntensity, setGlowIntensity] = useState(15);
  const [lines, setLines] = useState<DrawingLine[]>([]);
  const [hands, setHands] = useState<{ pos: Point, gesture: Gesture, id: number }[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isGrabbing, setIsGrabbing] = useState(false);
  const [isErasing, setIsErasing] = useState(false);
  const [bgOpacity, setBgOpacity] = useState(0.8);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Smoothing state
  const smoothedHandsRef = useRef<{ [key: number]: Point }>({});
  const SMOOTHING_FACTOR = 0.35; // Lower = smoother, more lag. Higher = snappier, more jitter.
  const gestureHistoryRef = useRef<{ [key: number]: Gesture[] }>({});
  const GESTURE_HISTORY_SIZE = 5;

  // For two-handed scaling/rotation
  const initialPinchDistRef = useRef<number | null>(null);
  const initialLinePointsRef = useRef<Point[] | null>(null);
  const initialAngleRef = useRef<number | null>(null);

  // Sync linesRef with state for the tracking loop
  useEffect(() => {
    linesRef.current = lines;
  }, [lines]);

  // --- Initialization ---

  useEffect(() => {
    async function initHandTracking() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.7,
          minHandPresenceConfidence: 0.7,
          minTrackingConfidence: 0.7
        });
        handLandmarkerRef.current = handLandmarker;
        setIsLoaded(true);
      } catch (error) {
        console.error("Failed to initialize hand tracking:", error);
      }
    }
    initHandTracking();
  }, []);

  useEffect(() => {
    if (isCameraOn && isLoaded) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isCameraOn, isLoaded]);

  const startCamera = async () => {
    if (!videoRef.current) return;
    try {
      setCameraError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' }
      });
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current?.play();
        requestRef.current = requestAnimationFrame(predictWebcam);
      };
    } catch (err) {
      console.error("Error accessing camera:", err);
      setCameraError(err instanceof Error ? err.message : String(err));
      setIsCameraOn(false);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (requestRef.current) {
      cancelAnimationFrame(requestRef.current);
    }
  };

  // --- Hand Tracking & Gesture Recognition ---

  const detectGesture = (landmarks: any[]): Gesture => {
    // MediaPipe landmarks: 0: wrist, 4: thumb tip, 8: index tip, 12: middle tip, 16: ring tip, 20: pinky tip
    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const middleTip = landmarks[12];
    const ringTip = landmarks[16];
    const pinkyTip = landmarks[20];
    const wrist = landmarks[0];

    // Distance helper
    const dist = (p1: any, p2: any) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

    // Hand scale factor (distance between wrist and middle finger base)
    const handScale = dist(wrist, landmarks[9]);

    // 1. Point: Index up, others down
    const isIndexUp = indexTip.y < landmarks[6].y;
    const isMiddleDown = middleTip.y > landmarks[10].y;
    const isRingDown = ringTip.y > landmarks[14].y;
    const isPinkyDown = pinkyTip.y > landmarks[18].y;
    
    if (isIndexUp && isMiddleDown && isRingDown && isPinkyDown) return 'POINT';

    // 2. Palm: All fingers up and spread
    const allUp = indexTip.y < landmarks[6].y && middleTip.y < landmarks[10].y && ringTip.y < landmarks[14].y && pinkyTip.y < landmarks[18].y;
    if (allUp && dist(indexTip, middleTip) > handScale * 0.4) return 'PALM';
    
    // 3. Pinch: Thumb and Index tips close together
    if (dist(thumbTip, indexTip) < handScale * 0.5) return 'PINCH';

    // 4. Fist: All fingers down
    const allDown = indexTip.y > landmarks[6].y && middleTip.y > landmarks[10].y && ringTip.y > landmarks[14].y && pinkyTip.y > landmarks[18].y;
    if (allDown) return 'FIST';

    return 'NONE';
  };

  const predictWebcam = async () => {
    if (!videoRef.current || !handLandmarkerRef.current || !canvasRef.current) return;

    const startTimeMs = performance.now();
    const results = handLandmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);

    const detectedHands: { pos: Point, gesture: Gesture, id: number }[] = [];

    if (results.landmarks && results.landmarks.length > 0) {
      results.landmarks.forEach((landmarks, index) => {
        const rawGesture = detectGesture(landmarks);
        
        // Gesture smoothing
        if (!gestureHistoryRef.current[index]) gestureHistoryRef.current[index] = [];
        gestureHistoryRef.current[index].push(rawGesture);
        if (gestureHistoryRef.current[index].length > GESTURE_HISTORY_SIZE) {
          gestureHistoryRef.current[index].shift();
        }
        
        // Find the most frequent gesture in history
        const counts: { [key: string]: number } = {};
        gestureHistoryRef.current[index].forEach(g => counts[g] = (counts[g] || 0) + 1);
        const gesture = (Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b) as Gesture);

        const rawX = (1 - landmarks[8].x) * canvasRef.current!.width;
        const rawY = landmarks[8].y * canvasRef.current!.height;
        
        // Apply exponential smoothing
        const prevPos = smoothedHandsRef.current[index] || { x: rawX, y: rawY };
        const smoothedX = prevPos.x + (rawX - prevPos.x) * SMOOTHING_FACTOR;
        const smoothedY = prevPos.y + (rawY - prevPos.y) * SMOOTHING_FACTOR;
        const pos = { x: smoothedX, y: smoothedY };
        
        smoothedHandsRef.current[index] = pos;
        detectedHands.push({ pos, gesture, id: index });
      });

      setHands(detectedHands);
      
      // Use the primary hand (index 0) for drawing/erasing/single-pinch
      // Use both hands for scaling if both are pinching
      handleMultiHandActions(detectedHands);
    } else {
      setHands([]);
      setIsGrabbing(false);
      setIsErasing(false);
      activeLineIdRef.current = null;
      movingLineIdRef.current = null;
      pinchStartPointRef.current = null;
      initialPinchDistRef.current = null;
    }

    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  const handleMultiHandActions = (detectedHands: { pos: Point, gesture: Gesture, id: number }[]) => {
    const hand1 = detectedHands[0];
    const hand2 = detectedHands[1];

    // Case 1: Two-Handed Move (Scale & Rotate)
    // Triggered if both hands are either PINCHING or FISTING
    const isHand1Grabbing = hand1 && (hand1.gesture === 'PINCH' || hand1.gesture === 'FIST');
    const isHand2Grabbing = hand2 && (hand2.gesture === 'PINCH' || hand2.gesture === 'FIST');

    if (isHand1Grabbing && isHand2Grabbing) {
      handleTwoHandedPinch(hand1.pos, hand2.pos);
      return;
    } else {
      initialPinchDistRef.current = null;
      initialLinePointsRef.current = null;
      initialAngleRef.current = null;
    }

    // Case 2: Single Hand Actions
    if (hand1) {
      handleGestureAction(hand1.gesture, hand1.pos);
    }
  };

  const handleTwoHandedPinch = (p1: Point, p2: Point) => {
    const dist = Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x);
    const center = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };

    if (!movingLineIdRef.current) {
      // Find closest line to the center of the two hands
      let closestId = null;
      let minDist = Infinity;
      linesRef.current.forEach(line => {
        line.points.forEach(p => {
          const d = Math.sqrt(Math.pow(p.x - center.x, 2) + Math.pow(p.y - center.y, 2));
          if (d < minDist && d < 120) {
            minDist = d;
            closestId = line.id;
          }
        });
      });

      if (closestId) {
        movingLineIdRef.current = closestId;
        initialPinchDistRef.current = dist;
        initialAngleRef.current = angle;
        const line = linesRef.current.find(l => l.id === closestId);
        if (line) initialLinePointsRef.current = [...line.points];
        setIsGrabbing(true);
      }
    } else if (initialLinePointsRef.current && initialPinchDistRef.current !== null && initialAngleRef.current !== null) {
      const scale = dist / initialPinchDistRef.current;
      const rotation = angle - initialAngleRef.current;

      // Calculate center of the original line points
      const linePoints = initialLinePointsRef.current;
      const lineCenter = linePoints.reduce((acc, p) => ({ x: acc.x + p.x / linePoints.length, y: acc.y + p.y / linePoints.length }), { x: 0, y: 0 });

      setLines(prev => prev.map(line => {
        if (line.id === movingLineIdRef.current) {
          const newPoints = linePoints.map(p => {
            // Translate to origin
            let tx = p.x - lineCenter.x;
            let ty = p.y - lineCenter.y;

            // Scale
            tx *= scale;
            ty *= scale;

            // Rotate
            const rx = tx * Math.cos(rotation) - ty * Math.sin(rotation);
            const ry = tx * Math.sin(rotation) + ty * Math.cos(rotation);

            // Translate back to the current pinch center
            return { x: rx + center.x, y: ry + center.y };
          });
          return { ...line, points: newPoints };
        }
        return line;
      }));
    }
  };

  // --- Drawing Logic ---

  const handleGestureAction = (gesture: Gesture, point: Point) => {
    if (gesture === 'POINT') {
      if (!activeLineIdRef.current) {
        const newLineId = Math.random().toString(36).substring(7);
        activeLineIdRef.current = newLineId;
        lastCreatedLineIdRef.current = newLineId;
        isRecentLineReadyRef.current = false; // Reset while drawing
        setLines(prev => [...prev, {
          id: newLineId,
          points: [point],
          color,
          thickness,
          glow: glowIntensity
        }]);
      } else {
        setLines(prev => prev.map(line => 
          line.id === activeLineIdRef.current 
            ? { ...line, points: [...line.points, point] }
            : line
        ));
      }
      movingLineIdRef.current = null;
      pinchStartPointRef.current = null;
      setIsGrabbing(false);
      setIsErasing(false);
    } else if (gesture === 'PALM') {
      setIsErasing(true);
      // Erase lines near the palm
      setLines(prev => prev.filter(line => {
        return !line.points.some(p => {
          const d = Math.sqrt(Math.pow(p.x - point.x, 2) + Math.pow(p.y - point.y, 2));
          return d < 100; // Large erase radius for palm
        });
      }));
      activeLineIdRef.current = null;
      setIsGrabbing(false);
    } else if (gesture === 'PINCH' || gesture === 'FIST') {
      setIsErasing(false);
      // Move logic (now triggered by PINCH or FIST)
      if (!movingLineIdRef.current) {
        let closestId = null;
        let minDist = Infinity;
        
        // 1. Try normal proximity grab
        linesRef.current.forEach(line => {
          line.points.forEach(p => {
            const d = Math.sqrt(Math.pow(p.x - point.x, 2) + Math.pow(p.y - point.y, 2));
            if (d < minDist && d < 80) {
              minDist = d;
              closestId = line.id;
            }
          });
        });

        // 2. If no proximity grab, check if we should grab the most recent line
        if (!closestId && isRecentLineReadyRef.current && lastCreatedLineIdRef.current) {
          closestId = lastCreatedLineIdRef.current;
        }

        if (closestId) {
          movingLineIdRef.current = closestId;
          pinchStartPointRef.current = point;
          setIsGrabbing(true);
          isRecentLineReadyRef.current = false; // Used it, so clear it
        }
      } else {
        // Offset points
        const dx = point.x - (pinchStartPointRef.current?.x || point.x);
        const dy = point.y - (pinchStartPointRef.current?.y || point.y);
        setLines(prev => prev.map(line => 
          line.id === movingLineIdRef.current 
            ? { ...line, points: line.points.map(p => ({ x: p.x + dx, y: p.y + dy })) }
            : line
        ));
        pinchStartPointRef.current = point;
      }
      activeLineIdRef.current = null;
    } else {
      // If we just finished drawing (POINT -> NONE/FIST), make it ready for pinch-selection
      if (activeLineIdRef.current) {
        isRecentLineReadyRef.current = true;
      }

      // Drop logic
      if (movingLineIdRef.current) {
        setIsGrabbing(false);
      }
      setIsErasing(false);
      activeLineIdRef.current = null;
      movingLineIdRef.current = null;
      pinchStartPointRef.current = null;
    }
  };

  // --- Canvas Rendering ---

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    window.addEventListener('resize', resize);
    resize();

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      lines.forEach(line => {
        if (line.points.length < 2) return;
        
        const isBeingMoved = line.id === movingLineIdRef.current;
        
        ctx.beginPath();
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = line.color;
        ctx.lineWidth = isBeingMoved ? line.thickness + 4 : line.thickness;
        
        // Glow effect
        ctx.shadowBlur = isBeingMoved ? line.glow + 20 : line.glow;
        ctx.shadowColor = line.color;

        // Smoother drawing using quadratic curves
        ctx.moveTo(line.points[0].x, line.points[0].y);
        
        let i;
        for (i = 1; i < line.points.length - 2; i++) {
          const xc = (line.points[i].x + line.points[i + 1].x) / 2;
          const yc = (line.points[i].y + line.points[i + 1].y) / 2;
          ctx.quadraticCurveTo(line.points[i].x, line.points[i].y, xc, yc);
        }
        
        // For the last 2 points
        if (line.points.length > 2) {
          ctx.quadraticCurveTo(
            line.points[i].x,
            line.points[i].y,
            line.points[i + 1].x,
            line.points[i + 1].y
          );
        } else {
          ctx.lineTo(line.points[1].x, line.points[1].y);
        }
        
        ctx.stroke();
        ctx.shadowBlur = 0;
      });

      requestAnimationFrame(render);
    };
    const animId = requestAnimationFrame(render);

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    };
  }, [lines]);

  // --- UI Handlers ---

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const clearAll = () => {
    setLines([]);
  };

  // --- Components ---

  const GestureIndicator = ({ type, active }: { type: Gesture, active: boolean }) => {
    const icons = {
      POINT: <MousePointer2 className="w-5 h-5" />,
      PALM: <Hand className="w-5 h-5" />,
      PINCH: <Grab className="w-5 h-5" />,
      FIST: <CircleStop className="w-5 h-5" />,
      NONE: null
    };

    if (type === 'NONE') return null;

    return (
      <div className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded-full transition-all duration-300",
        active ? "bg-white/20 text-white scale-110" : "bg-white/5 text-white/40"
      )}>
        {icons[type]}
        <span className="text-xs font-medium uppercase tracking-wider">{type}</span>
      </div>
    );
  };

  return (
    <div className="relative w-screen h-screen bg-black overflow-hidden font-sans">
      {/* Background Video */}
      <video
        ref={videoRef}
        className={cn(
          "absolute inset-0 w-full h-full object-cover transition-opacity duration-700",
          isCameraOn ? "scale-x-[-1]" : "opacity-0"
        )}
        style={{ opacity: isCameraOn ? bgOpacity : 0 }}
        autoPlay
        playsInline
        muted
      />

      {/* Drawing Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full z-10"
      />

      {/* Hand Cursors (Visual Feedback) */}
      {hands.map((hand) => (
        <motion.div
          key={hand.id}
          className="absolute z-20 pointer-events-none"
          animate={{ 
            x: hand.pos.x - (hand.gesture === 'PALM' ? 50 : 12), 
            y: hand.pos.y - (hand.gesture === 'PALM' ? 50 : 12),
            scale: (hand.gesture === 'PINCH' || hand.gesture === 'FIST') ? 1.5 : 1
          }}
          transition={{ type: 'spring', damping: 25, stiffness: 400, mass: 0.5 }}
        >
          {hand.gesture === 'PALM' ? (
            <div className="w-[100px] h-[100px] rounded-full border-2 border-white/80 bg-white/10 flex items-center justify-center shadow-[0_0_40px_rgba(255,255,255,0.6)]">
              <Eraser className="w-10 h-10 text-white animate-pulse" />
            </div>
          ) : (
            <div 
              className={cn(
                "w-6 h-6 rounded-full border-2 flex items-center justify-center transition-colors duration-300",
                (hand.gesture === 'PINCH' || hand.gesture === 'FIST') ? "border-white bg-white/20" : "border-white/50 bg-transparent"
              )}
              style={{ boxShadow: `0 0 ${(hand.gesture === 'PINCH' || hand.gesture === 'FIST') ? '25px' : '15px'} ${color}` }}
            >
              {(hand.gesture === 'PINCH' || hand.gesture === 'FIST') ? (
                <Grab className="w-3 h-3 text-white" />
              ) : (
                <div className="w-1.5 h-1.5 rounded-full bg-white" />
              )}
            </div>
          )}
        </motion.div>
      ))}

      {/* Glass Sidebar Menu */}
      <div className="absolute left-6 top-1/2 -translate-y-1/2 z-30 flex flex-col gap-6">
        <motion.div 
          initial={{ x: -100, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="glass-panel p-4 rounded-3xl flex flex-col gap-6 w-56"
        >
          {/* Color Palette */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2 text-white/60 mb-1">
              <Palette className="w-4 h-4" />
              <span className="text-[10px] font-bold uppercase tracking-widest">Colors</span>
            </div>
            <div className="grid grid-cols-4 gap-2">
              {NEON_COLORS.map((c) => (
                <button
                  key={c.value}
                  onClick={() => setColor(c.value)}
                  className={cn(
                    "w-8 h-8 rounded-xl transition-all duration-300 hover:scale-110 active:scale-95",
                    color === c.value ? "ring-2 ring-white ring-offset-2 ring-offset-black/50 scale-110" : "opacity-60"
                  )}
                  style={{ backgroundColor: c.value, boxShadow: color === c.value ? `0 0 15px ${c.value}` : 'none' }}
                />
              ))}
            </div>
          </div>

          <div className="h-px bg-white/10 w-full" />

          {/* Thickness Slider */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between text-white/60 mb-1">
              <div className="flex items-center gap-2">
                <Type className="w-4 h-4" />
                <span className="text-[10px] font-bold uppercase tracking-widest">Size</span>
              </div>
              <span className="text-[10px] font-mono">{thickness}px</span>
            </div>
            <input
              type="range"
              min="1"
              max="30"
              value={thickness}
              onChange={(e) => setThickness(parseInt(e.target.value))}
              className="w-full accent-white opacity-70 hover:opacity-100 transition-opacity"
            />
          </div>

          <div className="h-px bg-white/10 w-full" />

          {/* Glow Slider */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between text-white/60 mb-1">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full border border-white/60 flex items-center justify-center">
                  <div className="w-1 h-1 bg-white/60 rounded-full" />
                </div>
                <span className="text-[10px] font-bold uppercase tracking-widest">Glow</span>
              </div>
              <span className="text-[10px] font-mono">{glowIntensity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="50"
              value={glowIntensity}
              onChange={(e) => setGlowIntensity(parseInt(e.target.value))}
              className="w-full accent-white opacity-70 hover:opacity-100 transition-opacity"
            />
          </div>

          <div className="h-px bg-white/10 w-full" />

          {/* Background Opacity Slider */}
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between text-white/60 mb-1">
              <div className="flex items-center gap-2">
                <Camera className="w-4 h-4" />
                <span className="text-[10px] font-bold uppercase tracking-widest">Camera</span>
              </div>
              <span className="text-[10px] font-mono">{Math.round(bgOpacity * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={bgOpacity * 100}
              onChange={(e) => setBgOpacity(parseInt(e.target.value) / 100)}
              className="w-full accent-white opacity-70 hover:opacity-100 transition-opacity"
            />
          </div>

          <div className="h-px bg-white/10 w-full" />

          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={clearAll}
              className="flex-1 p-3 rounded-2xl bg-red-500/10 text-red-500 hover:bg-red-500/20 transition-colors flex items-center justify-center"
              title="Clear All"
            >
              <Trash2 className="w-5 h-5" />
            </button>
            <button
              onClick={() => setIsCameraOn(!isCameraOn)}
              className={cn(
                "flex-1 p-3 rounded-2xl transition-colors flex items-center justify-center",
                isCameraOn ? "bg-white/10 text-white hover:bg-white/20" : "bg-white/5 text-white/40"
              )}
              title="Toggle Camera"
            >
              {isCameraOn ? <Camera className="w-5 h-5" /> : <CameraOff className="w-5 h-5" />}
            </button>
            <button
              onClick={toggleFullscreen}
              className="flex-1 p-3 rounded-2xl bg-white/10 text-white hover:bg-white/20 transition-colors flex items-center justify-center"
              title="Fullscreen"
            >
              {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
            </button>
          </div>
        </motion.div>
      </div>

      {/* Bottom Gesture Status */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 z-30 flex gap-4">
        <motion.div 
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="glass-panel px-6 py-3 rounded-full flex items-center gap-6"
        >
          <GestureIndicator type="POINT" active={hands[0]?.gesture === 'POINT'} />
          <GestureIndicator type="PALM" active={hands[0]?.gesture === 'PALM'} />
          <GestureIndicator type="PINCH" active={hands[0]?.gesture === 'PINCH' || hands[1]?.gesture === 'PINCH' || hands[0]?.gesture === 'FIST' || hands[1]?.gesture === 'FIST'} />
        </motion.div>
      </div>

      {/* Loading Overlay */}
      <AnimatePresence>
        {!isLoaded && (
          <motion.div 
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 bg-black flex flex-col items-center justify-center gap-6"
          >
            <div className="relative">
              <div className="w-24 h-24 border-4 border-white/10 rounded-full" />
              <div className="absolute inset-0 w-24 h-24 border-4 border-t-white rounded-full animate-spin" />
            </div>
            <div className="flex flex-col items-center gap-2">
              <h2 className="text-2xl font-bold tracking-tighter">Initializing Magic Marker</h2>
              <p className="text-white/40 text-sm">Loading hand tracking models...</p>
            </div>
          </motion.div>
        )}

        {cameraError && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute inset-0 z-40 bg-black/90 backdrop-blur-xl flex flex-col items-center justify-center gap-6 p-8 text-center"
          >
            <div className="w-20 h-20 rounded-full bg-red-500/20 flex items-center justify-center mb-4">
              <CameraOff className="w-10 h-10 text-red-500" />
            </div>
            <div className="max-w-md flex flex-col gap-4">
              <h2 className="text-3xl font-bold tracking-tight text-white">Camera Access Required</h2>
              <p className="text-white/60 leading-relaxed">
                Magic Marker needs camera access to track your hand gestures. 
                Please ensure you've granted permission in your browser settings.
              </p>
              <div className="bg-white/5 p-4 rounded-2xl border border-white/10 text-xs font-mono text-red-400 text-left">
                Error: {cameraError}
              </div>
            </div>
            <button
              onClick={() => {
                setIsCameraOn(true);
                startCamera();
              }}
              className="mt-4 px-8 py-4 rounded-2xl bg-white text-black font-bold hover:bg-white/90 transition-all active:scale-95 flex items-center gap-2"
            >
              <Camera className="w-5 h-5" />
              Try Again
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top Info */}
      <div className="absolute top-8 left-1/2 -translate-x-1/2 z-30 text-center pointer-events-none">
        <h1 className="text-xl font-black tracking-[0.2em] uppercase text-white/20">Magic Marker</h1>
      </div>
    </div>
  );
}
