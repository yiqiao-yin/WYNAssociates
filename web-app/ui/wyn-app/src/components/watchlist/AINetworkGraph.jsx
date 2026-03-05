import { useEffect, useRef, useCallback, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { aiWatchlistData } from '../../data/aiWatchlistData';
import {
  LAYER_ORDER, LAYER_COLORS, TICKER_NAMES, EDGES,
  PRIVATE_COMPANIES, CONFIDENCE_META,
} from '../../data/aiRelationships';

function ticker(symbol) {
  return symbol.includes(':') ? symbol.split(':')[1] : symbol;
}

// Inverted radius: Energy (bottom) = largest, Application (top) = smallest
function layerRadius(layerIdx) {
  return [14, 11.5, 9, 7, 5][layerIdx] || 8;
}

// Confidence -> visual parameters
const CONF = {
  high:   { opacity: 0.45, highlightOpacity: 1.0, lineWidth: 1, beamSpeed: 0.008, beamBright: 1.0 },
  medium: { opacity: 0.22, highlightOpacity: 0.7, lineWidth: 1, beamSpeed: 0.005, beamBright: 0.6 },
  low:    { opacity: 0.10, highlightOpacity: 0.4, lineWidth: 1, beamSpeed: 0.003, beamBright: 0.35 },
};

function buildNodes() {
  const nodes = new Map();
  const layerSpacing = 6;

  // Public companies from aiWatchlistData
  LAYER_ORDER.forEach((layerName, layerIdx) => {
    const symbols = aiWatchlistData[layerName] || [];
    const y = layerIdx * layerSpacing;
    const radius = layerRadius(layerIdx);
    symbols.forEach((sym, i) => {
      const t = ticker(sym);
      const angle = (i / symbols.length) * Math.PI * 2;
      if (!nodes.has(t)) {
        nodes.set(t, {
          ticker: t, layer: layerName, layerIdx, isPrivate: false,
          position: new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius),
          color: LAYER_COLORS[layerName],
        });
      }
    });
  });

  // Private companies: insert into their layer rings
  const privateByLayer = {};
  Object.entries(PRIVATE_COMPANIES).forEach(([tk, layer]) => {
    if (!privateByLayer[layer]) privateByLayer[layer] = [];
    privateByLayer[layer].push(tk);
  });

  LAYER_ORDER.forEach((layerName, layerIdx) => {
    const privates = privateByLayer[layerName] || [];
    if (!privates.length) return;
    const publicCount = (aiWatchlistData[layerName] || []).length;
    const y = layerIdx * layerSpacing;
    const radius = layerRadius(layerIdx) * 0.7; // inner ring for private companies
    privates.forEach((tk, i) => {
      if (nodes.has(tk)) return;
      const angle = (i / privates.length) * Math.PI * 2 + 0.3; // slight offset
      nodes.set(tk, {
        ticker: tk, layer: layerName, layerIdx, isPrivate: true,
        position: new THREE.Vector3(Math.cos(angle) * radius, y, Math.sin(angle) * radius),
        color: LAYER_COLORS[layerName],
      });
    });
  });

  return nodes;
}

function buildEdges(nodes) {
  const validEdges = [];
  EDGES.forEach(({ from, to, reason, confidence }) => {
    if (from === to) return;
    const srcNode = nodes.get(from);
    const tgtNode = nodes.get(to);
    if (srcNode && tgtNode) {
      validEdges.push({ from: srcNode, to: tgtNode, reason, confidence: confidence || 'medium' });
    }
  });
  return validEdges;
}

const btnStyle = (active) => ({
  background: active ? '#555' : 'rgba(30,30,30,0.85)',
  border: '1px solid #555',
  color: active ? '#fff' : '#bbb',
  padding: '5px 10px', borderRadius: 4, cursor: 'pointer',
  fontSize: 13, fontFamily: 'Arial, sans-serif',
  display: 'flex', alignItems: 'center', gap: 4,
});

const checkStyle = (active) => ({
  ...btnStyle(active),
  padding: '4px 8px', fontSize: 11,
});

export default function AINetworkGraph() {
  const mountRef = useRef(null);
  const tooltipRef = useRef(null);
  const cleanupRef = useRef(null);
  const stateRef = useRef({ spinning: true, camera: null, controls: null });
  const edgeGroupRef = useRef(null);
  const edgesDataRef = useRef([]);

  const [spinning, setSpinning] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [confFilter, setConfFilter] = useState({ high: true, medium: true, low: true });
  const [loadProgress, setLoadProgress] = useState(0); // 0-100
  const [loadStage, setLoadStage] = useState('Initializing...');
  const [loaded, setLoaded] = useState(false);
  const wrapperRef = useRef(null);

  useEffect(() => { stateRef.current.spinning = spinning; }, [spinning]);

  // Sync confidence filter to Three.js lines
  useEffect(() => {
    const edgeGroup = edgeGroupRef.current;
    const edgesData = edgesDataRef.current;
    if (!edgeGroup) return;
    edgesData.forEach(({ line, confidence }) => {
      const visible = confFilter[confidence];
      line.visible = visible;
    });
  }, [confFilter]);

  const handleFullscreen = useCallback(() => {
    const el = wrapperRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', onFsChange);
    return () => document.removeEventListener('fullscreenchange', onFsChange);
  }, []);

  const handleZoom = useCallback((dir) => {
    const { camera, controls } = stateRef.current;
    if (!camera || !controls) return;
    const vec = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
    camera.position.addScaledVector(vec, dir * -3);
  }, []);

  const toggleConf = useCallback((level) => {
    setConfFilter((prev) => ({ ...prev, [level]: !prev[level] }));
  }, []);

  // Yield to browser so it can paint progress updates between heavy stages
  const nextFrame = () => new Promise((r) => requestAnimationFrame(r));

  const init = useCallback(async () => {
    const container = mountRef.current;
    if (!container) return;

    // ── Stage 1: WebGL setup (10%) ──
    setLoadStage('Setting up WebGL...');
    setLoadProgress(5);
    await nextFrame();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0a);
    scene.fog = new THREE.FogExp2(0x0a0a0a, 0.005);

    const width = container.clientWidth;
    const height = container.clientHeight;
    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 200);
    camera.position.set(22, 16, 24);
    camera.lookAt(0, 12, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 12, 0);
    controls.minDistance = 5;
    controls.maxDistance = 80;

    stateRef.current.camera = camera;
    stateRef.current.controls = controls;

    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const point = new THREE.PointLight(0xffffff, 1.2, 100);
    point.position.set(10, 30, 15);
    scene.add(point);

    setLoadProgress(15);

    // ── Stage 2: Build node data (20%) ──
    setLoadStage('Building node graph...');
    await nextFrame();

    const nodes = buildNodes();
    const edges = buildEdges(nodes);

    setLoadProgress(25);

    // ── Stage 3: Create node meshes (40%) ──
    setLoadStage(`Creating ${nodes.size} company nodes...`);
    await nextFrame();

    const nodeGroup = new THREE.Group();
    const nodeMeshes = [];
    const sphereGeo = new THREE.SphereGeometry(0.45, 16, 16);
    const diamondGeo = new THREE.OctahedronGeometry(0.5, 0);

    nodes.forEach((node) => {
      const mat = new THREE.MeshStandardMaterial({
        color: node.color,
        emissive: node.color,
        emissiveIntensity: 0.3,
        roughness: 0.4,
        metalness: 0.6,
        wireframe: node.isPrivate,
      });
      const geo = node.isPrivate ? diamondGeo : sphereGeo;
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(node.position);
      mesh.userData = node;
      nodeGroup.add(mesh);
      nodeMeshes.push(mesh);

      if (node.isPrivate) {
        const ringGeo = new THREE.RingGeometry(0.55, 0.7, 16);
        const ringMat = new THREE.MeshBasicMaterial({
          color: node.color, transparent: true, opacity: 0.3, side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.position.copy(node.position);
        ring.lookAt(camera.position);
        nodeGroup.add(ring);
      }
    });
    scene.add(nodeGroup);

    setLoadProgress(40);

    // ── Stage 4: Ticker labels (55%) ──
    setLoadStage('Rendering ticker labels...');
    await nextFrame();

    nodes.forEach((node) => {
      const canvas = document.createElement('canvas');
      canvas.width = 256;
      canvas.height = 48;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = node.isPrivate ? '#ffdd57' : '#ffffff';
      ctx.font = node.isPrivate ? 'italic bold 22px Arial' : 'bold 24px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(node.ticker, 128, 30);
      const tex = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.75 });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(node.position);
      sprite.position.y += 0.85;
      sprite.scale.set(2.8, 0.5, 1);
      nodeGroup.add(sprite);
    });

    // Layer labels
    LAYER_ORDER.forEach((layerName, idx) => {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 64;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = LAYER_COLORS[layerName];
      ctx.font = 'bold 36px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(layerName.toUpperCase(), 256, 42);
      const tex = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.85 });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.set(0, idx * 6 - 1.5, 0);
      sprite.scale.set(10, 1.2, 1);
      scene.add(sprite);
    });

    // Layer rings
    LAYER_ORDER.forEach((layerName, idx) => {
      const radius = layerRadius(idx);
      const ringGeo = new THREE.RingGeometry(radius - 0.05, radius + 0.05, 64);
      const ringMat = new THREE.MeshBasicMaterial({
        color: LAYER_COLORS[layerName], transparent: true, opacity: 0.2, side: THREE.DoubleSide,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = idx * 6;
      scene.add(ring);

      if (LAYER_ORDER[idx] && Object.values(PRIVATE_COMPANIES).includes(layerName)) {
        const innerR = radius * 0.7;
        const innerGeo = new THREE.RingGeometry(innerR - 0.03, innerR + 0.03, 64);
        const innerMat = new THREE.MeshBasicMaterial({
          color: LAYER_COLORS[layerName], transparent: true, opacity: 0.1, side: THREE.DoubleSide,
        });
        const innerRing = new THREE.Mesh(innerGeo, innerMat);
        innerRing.rotation.x = -Math.PI / 2;
        innerRing.position.y = idx * 6;
        scene.add(innerRing);
      }
    });

    setLoadProgress(55);

    // ── Stage 5: Edges (75%) ──
    setLoadStage(`Drawing ${edges.length} supply-chain edges...`);
    await nextFrame();

    const edgeGroup = new THREE.Group();
    const edgeCurves = [];
    edgeGroupRef.current = edgeGroup;

    edges.forEach(({ from, to, reason, confidence }) => {
      const conf = CONF[confidence] || CONF.medium;
      const start = from.position.clone();
      const end = to.position.clone();
      const mid = start.clone().lerp(end, 0.5);
      mid.x += (Math.random() - 0.5) * 1.5;
      mid.z += (Math.random() - 0.5) * 1.5;

      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      const curvePoints = curve.getPoints(30);
      const geo = new THREE.BufferGeometry().setFromPoints(curvePoints);

      let mat;
      if (confidence === 'low') {
        const lineGeo = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const distances = [0];
        for (let i = 1; i < curvePoints.length; i++) {
          distances.push(distances[i - 1] + curvePoints[i].distanceTo(curvePoints[i - 1]));
        }
        lineGeo.setAttribute('lineDistance', new THREE.Float32BufferAttribute(distances, 1));
        mat = new THREE.LineDashedMaterial({
          color: from.color, transparent: true, opacity: conf.opacity,
          dashSize: 0.3, gapSize: 0.2,
        });
        const line = new THREE.Line(lineGeo, mat);
        line.computeLineDistances();
        line.userData = { from, to, reason, confidence };
        edgeGroup.add(line);
        edgeCurves.push({ curve, line, from, to, confidence });
      } else {
        mat = new THREE.LineBasicMaterial({
          color: from.color, transparent: true, opacity: conf.opacity,
        });
        const line = new THREE.Line(geo, mat);
        line.userData = { from, to, reason, confidence };
        edgeGroup.add(line);
        edgeCurves.push({ curve, line, from, to, confidence });
      }
    });
    scene.add(edgeGroup);
    edgesDataRef.current = edgeCurves;

    setLoadProgress(75);

    // ── Stage 6: Lightbeam pool (90%) ──
    setLoadStage('Initializing lightbeam effects...');
    await nextFrame();

    const BEAM_POOL_SIZE = 150;
    const beamGroup = new THREE.Group();
    const beamGeo = new THREE.SphereGeometry(0.12, 8, 8);
    const beams = [];

    for (let i = 0; i < BEAM_POOL_SIZE; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 });
      const mesh = new THREE.Mesh(beamGeo, mat);
      mesh.visible = false;
      beamGroup.add(mesh);
      beams.push({ mesh, t: 0, speed: 0, curve: null, active: false, brightness: 1 });
    }
    scene.add(beamGroup);

    setLoadProgress(90);

    // ── Stage 7: Event handlers & animation (100%) ──
    setLoadStage('Starting render loop...');
    await nextFrame();

    function activateBeams(hoveredTicker) {
      const relevant = edgeCurves.filter(
        (ec) => ec.line.visible && (ec.from.ticker === hoveredTicker || ec.to.ticker === hoveredTicker)
      );
      let bi = 0;
      relevant.forEach((ec) => {
        const conf = CONF[ec.confidence] || CONF.medium;
        const perEdge = Math.max(2, Math.floor(BEAM_POOL_SIZE / Math.max(relevant.length, 1)));
        for (let j = 0; j < perEdge && bi < BEAM_POOL_SIZE; j++) {
          const b = beams[bi++];
          b.active = true;
          b.curve = ec.curve;
          b.t = j / perEdge;
          b.speed = conf.beamSpeed + Math.random() * 0.003;
          b.brightness = conf.beamBright;
          b.mesh.material.color.set(ec.from.color);
          b.mesh.visible = true;
        }
      });
      for (; bi < BEAM_POOL_SIZE; bi++) {
        beams[bi].active = false;
        beams[bi].mesh.visible = false;
      }
    }

    function deactivateBeams() {
      beams.forEach((b) => { b.active = false; b.mesh.visible = false; });
    }

    function updateBeams() {
      const rotY = nodeGroup.rotation.y;
      const cos = Math.cos(rotY);
      const sin = Math.sin(rotY);
      beams.forEach((b) => {
        if (!b.active || !b.curve) return;
        b.t += b.speed;
        if (b.t > 1) b.t -= 1;
        const pt = b.curve.getPoint(b.t);
        b.mesh.position.set(pt.x * cos + pt.z * sin, pt.y, -pt.x * sin + pt.z * cos);
        const fade = Math.sin(b.t * Math.PI);
        b.mesh.material.opacity = b.brightness * 0.9 * fade;
        b.mesh.scale.setScalar(0.8 + fade * 0.5);
      });
    }

    // Raycaster
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    let hoveredNode = null;

    function resetHighlights() {
      nodeMeshes.forEach((m) => { m.material.emissiveIntensity = 0.3; });
      edgeGroup.children.forEach((l) => {
        const c = CONF[l.userData.confidence] || CONF.medium;
        l.material.opacity = c.opacity;
      });
      deactivateBeams();
    }

    function onMouseMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(nodeMeshes);
      const tooltip = tooltipRef.current;

      if (intersects.length > 0) {
        const hit = intersects[0].object;
        const node = hit.userData;

        if (hoveredNode !== hit) {
          if (hoveredNode) {
            hoveredNode.scale.setScalar(1);
            resetHighlights();
          }

          hoveredNode = hit;
          hit.material.emissiveIntensity = 1.0;
          hit.scale.setScalar(1.6);

          const connected = new Set([node.ticker]);
          edgeGroup.children.forEach((l) => {
            if (!l.visible) return;
            const ef = l.userData.from.ticker;
            const et = l.userData.to.ticker;
            if (ef === node.ticker || et === node.ticker) {
              const c = CONF[l.userData.confidence] || CONF.medium;
              l.material.opacity = c.highlightOpacity;
              connected.add(ef);
              connected.add(et);
            } else {
              l.material.opacity = 0.03;
            }
          });

          nodeMeshes.forEach((m) => {
            m.material.emissiveIntensity = connected.has(m.userData.ticker) ? 0.7 : 0.06;
          });
          hit.material.emissiveIntensity = 1.0;
          activateBeams(node.ticker);
        }

        if (tooltip) {
          const name = TICKER_NAMES[node.ticker] || node.ticker;
          const priv = node.isPrivate ? ' <span style="color:#ffdd57;font-size:10px">[PRIVATE]</span>' : '';
          const connections = edges.filter(
            (e) => e.from.ticker === node.ticker || e.to.ticker === node.ticker
          );
          const inbound = connections
            .filter((e) => e.to.ticker === node.ticker)
            .map((e) => {
              const cm = CONFIDENCE_META[e.confidence] || CONFIDENCE_META.medium;
              return `<span style="color:${e.from.color}">${e.from.ticker}</span>: ${e.reason} <span style="color:${cm.color};font-size:9px">[${cm.label}]</span>`;
            }).slice(0, 6);
          const outbound = connections
            .filter((e) => e.from.ticker === node.ticker)
            .map((e) => {
              const cm = CONFIDENCE_META[e.confidence] || CONFIDENCE_META.medium;
              return `<span style="color:${e.to.color}">${e.to.ticker}</span>: ${e.reason} <span style="color:${cm.color};font-size:9px">[${cm.label}]</span>`;
            }).slice(0, 6);

          let html = `<strong style="font-size:14px">${node.ticker}</strong> <span style="color:${node.color}">${name}</span>${priv}`;
          html += `<br><span style="color:${node.color};font-size:11px">${node.layer}</span>`;
          if (inbound.length) {
            html += `<br><br><span style="color:#888;font-size:11px">RECEIVES FROM:</span>`;
            inbound.forEach((s) => { html += `<br><span style="font-size:11px">${s}</span>`; });
          }
          if (outbound.length) {
            html += `<br><br><span style="color:#888;font-size:11px">SUPPLIES TO:</span>`;
            outbound.forEach((s) => { html += `<br><span style="font-size:11px">${s}</span>`; });
          }
          tooltip.innerHTML = html;
          tooltip.style.display = 'block';
          tooltip.style.left = Math.min(event.clientX + 12, window.innerWidth - 360) + 'px';
          tooltip.style.top = Math.min(event.clientY - 10, window.innerHeight - 300) + 'px';
        }
      } else {
        if (hoveredNode) {
          hoveredNode.scale.setScalar(1);
          hoveredNode = null;
          resetHighlights();
        }
        if (tooltip) tooltip.style.display = 'none';
      }
    }

    renderer.domElement.addEventListener('mousemove', onMouseMove);

    renderer.domElement.addEventListener('click', () => {
      container.dispatchEvent(new CustomEvent('toggle-spin'));
    });
    function onToggleSpin() {
      stateRef.current.spinning = !stateRef.current.spinning;
      setSpinning(stateRef.current.spinning);
    }
    container.addEventListener('toggle-spin', onToggleSpin);

    let animId;
    function animate() {
      animId = requestAnimationFrame(animate);
      if (stateRef.current.spinning && !controls._isDragging) {
        nodeGroup.rotation.y += 0.0015;
        edgeGroup.rotation.y += 0.0015;
      }
      updateBeams();
      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    function onResize() {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }
    window.addEventListener('resize', onResize);

    cleanupRef.current = () => {
      cancelAnimationFrame(animId);
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      container.removeEventListener('toggle-spin', onToggleSpin);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      renderer.dispose();
      if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement);
    };

    setLoadProgress(100);
    setLoadStage('Ready');
    await nextFrame();
    setLoaded(true);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    // If the container is visible (has dimensions), init immediately.
    // Otherwise, wait for it to become visible via ResizeObserver.
    const tryInit = () => {
      if (container.clientWidth > 0 && container.clientHeight > 0) {
        init();
        return true;
      }
      return false;
    };

    if (tryInit()) {
      return () => { if (cleanupRef.current) cleanupRef.current(); };
    }

    // Container is hidden (display:none parent). Wait for it to get dimensions.
    const ro = new ResizeObserver(() => {
      if (tryInit()) {
        ro.disconnect();
      }
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      if (cleanupRef.current) cleanupRef.current();
    };
  }, [init]);

  return (
    <div ref={wrapperRef} style={{ position: 'relative', background: '#0a0a0a' }}>
      {/* Loading overlay */}
      {!loaded && (
        <div style={{
          position: 'absolute', inset: 0, zIndex: 20,
          background: '#0a0a0a',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          gap: 12,
        }}>
          <div style={{ color: '#888', fontSize: 13, fontFamily: 'Arial, sans-serif' }}>
            {loadStage}
          </div>
          {/* Progress bar */}
          <div style={{
            width: 260, height: 6, background: '#222', borderRadius: 3, overflow: 'hidden',
          }}>
            <div style={{
              width: `${loadProgress}%`, height: '100%',
              background: 'linear-gradient(90deg, #3b82f6, #a855f7, #22c55e)',
              borderRadius: 3,
              transition: 'width 0.2s ease-out',
            }} />
          </div>
          <div style={{ color: '#555', fontSize: 11, fontFamily: 'Arial, sans-serif' }}>
            {loadProgress}%
          </div>
        </div>
      )}

      {/* Legend */}
      <div style={{
        position: 'absolute', top: 8, left: 8, zIndex: 10,
        background: 'rgba(0,0,0,0.8)', padding: '10px 14px', borderRadius: 6,
        fontSize: 12, lineHeight: '20px', pointerEvents: 'none',
      }}>
        {LAYER_ORDER.slice().reverse().map((name) => (
          <div key={name}>
            <span style={{
              display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
              background: LAYER_COLORS[name], marginRight: 6, verticalAlign: 'middle',
            }} />
            {name}
          </div>
        ))}
        <div style={{ marginTop: 8, borderTop: '1px solid #333', paddingTop: 6 }}>
          <div><span style={{ color: '#fff', marginRight: 4 }}>&#9679;</span> Public company</div>
          <div><span style={{ color: '#ffdd57', marginRight: 4 }}>&#9670;</span> Private company</div>
        </div>
        <div style={{ marginTop: 6, color: '#555', fontSize: 10 }}>
          Click canvas to pause/resume
        </div>
      </div>

      {/* Toolbar */}
      <div style={{
        position: 'absolute', top: 8, right: 8, zIndex: 10,
        display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: 'flex-end',
      }}>
        <button
          onClick={() => { stateRef.current.spinning = !spinning; setSpinning(!spinning); }}
          style={btnStyle(spinning)}
          title={spinning ? 'Pause rotation' : 'Resume rotation'}
        >
          {spinning ? '\u23F8 Pause' : '\u25B6 Spin'}
        </button>
        <button onClick={() => handleZoom(1)} style={btnStyle(false)} title="Zoom in">+ Zoom</button>
        <button onClick={() => handleZoom(-1)} style={btnStyle(false)} title="Zoom out">- Zoom</button>
        <button onClick={handleFullscreen} style={btnStyle(isFullscreen)} title="Toggle fullscreen">
          {isFullscreen ? '\u2716 Exit' : '\u26F6 Full'}
        </button>
      </div>

      {/* Confidence filter */}
      <div style={{
        position: 'absolute', bottom: 8, left: 8, zIndex: 10,
        display: 'flex', gap: 4, background: 'rgba(0,0,0,0.7)',
        padding: '6px 10px', borderRadius: 6,
      }}>
        <span style={{ color: '#888', fontSize: 11, alignSelf: 'center', marginRight: 4 }}>Confidence:</span>
        {['high', 'medium', 'low'].map((level) => (
          <button
            key={level}
            onClick={() => toggleConf(level)}
            style={{
              ...checkStyle(confFilter[level]),
              borderColor: CONFIDENCE_META[level].color,
              color: confFilter[level] ? CONFIDENCE_META[level].color : '#555',
            }}
          >
            {confFilter[level] ? '\u2713 ' : '\u2717 '}
            {CONFIDENCE_META[level].label}
          </button>
        ))}
      </div>

      {/* 3D Canvas */}
      <div
        ref={mountRef}
        style={{ width: '100%', height: isFullscreen ? '100vh' : '70vh', minHeight: 400, cursor: 'grab' }}
      />

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        style={{
          display: 'none', position: 'fixed', zIndex: 9999,
          background: 'rgba(10,10,10,0.95)', color: '#eee',
          padding: '10px 14px', borderRadius: 6,
          border: '1px solid #444', maxWidth: 350,
          fontFamily: 'Arial, sans-serif', fontSize: 12, lineHeight: '16px',
          pointerEvents: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
        }}
      />
    </div>
  );
}
