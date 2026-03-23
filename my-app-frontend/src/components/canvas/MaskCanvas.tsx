'use client';
import { useRef, useState, useEffect, useCallback } from 'react';

interface MaskCanvasProps {
  imageFile: File;
  onMaskReady: (blob: Blob) => void;
}

export default function MaskCanvas({ imageFile, onMaskReady }: MaskCanvasProps) {
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing]   = useState(false);
  const [brushSize, setBrushSize] = useState(20);
  const [tool, setTool]         = useState<'brush' | 'eraser'>('brush');
  const [imgSize, setImgSize]   = useState({ w: 0, h: 0 });
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Load image onto display canvas
  useEffect(() => {
    const url = URL.createObjectURL(imageFile);
    const img = new Image();
    img.onload = () => {
      imageRef.current = img;
      const maxW = 600, maxH = 500;
      const ratio = Math.min(maxW / img.width, maxH / img.height, 1);
      const w = Math.round(img.width * ratio);
      const h = Math.round(img.height * ratio);
      setImgSize({ w, h });

      // Draw image on display canvas
      const canvas = canvasRef.current!;
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0, w, h);

      // Clear mask canvas
      const mask = maskCanvasRef.current!;
      mask.width = w; mask.height = h;
      const mctx = mask.getContext('2d')!;
      mctx.fillStyle = 'black';
      mctx.fillRect(0, 0, w, h);
    };
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [imageFile]);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const canvas = canvasRef.current!;
    const rect   = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    if ('touches' in e) {
      return {
        x: (e.touches[0].clientX - rect.left) * scaleX,
        y: (e.touches[0].clientY - rect.top)  * scaleY,
      };
    }
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    };
  };

  const draw = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    if (!drawing) return;
    const { x, y } = getPos(e);

    // Draw on display canvas (semi-transparent red overlay)
    const ctx = canvasRef.current!.getContext('2d')!;
    ctx.globalCompositeOperation = tool === 'brush' ? 'source-over' : 'source-over';
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fillStyle = tool === 'brush' ? 'rgba(108,99,255,0.45)' : 'rgba(0,0,0,0)';
    if (tool === 'eraser') {
      // Redraw image patch under eraser
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(imageRef.current!, 0, 0, canvasRef.current!.width, canvasRef.current!.height);
      ctx.restore();
    } else {
      ctx.fill();
    }

    // Update mask canvas (white = hole, black = known)
    const mctx = maskCanvasRef.current!.getContext('2d')!;
    mctx.beginPath();
    mctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    mctx.fillStyle = tool === 'brush' ? 'white' : 'black';
    mctx.fill();
  }, [drawing, brushSize, tool]);

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    setDrawing(true);
    draw(e);
  };

  const stopDraw = () => {
    if (!drawing) return;
    setDrawing(false);
    // Export mask as blob
    maskCanvasRef.current!.toBlob(blob => {
      if (blob) onMaskReady(blob);
    }, 'image/png');
  };

  const clearMask = () => {
    // Redraw original image
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(imageRef.current!, 0, 0, canvas.width, canvas.height);
    // Clear mask
    const mctx = maskCanvasRef.current!.getContext('2d')!;
    mctx.fillStyle = 'black';
    mctx.fillRect(0, 0, canvas.width, canvas.height);
    onMaskReady(new Blob());
  };

  return (
    <div>
      {/* Toolbar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: '12px',
        marginBottom: '12px', flexWrap: 'wrap',
      }}>
        {/* Tool buttons */}
        {(['brush', 'eraser'] as const).map(t => (
          <button key={t} onClick={() => setTool(t)} style={{
            padding: '6px 14px', borderRadius: '6px', fontSize: '13px',
            fontWeight: 500, cursor: 'pointer', border: '1px solid',
            borderColor: tool === t ? 'var(--accent)' : 'var(--border)',
            background:  tool === t ? 'rgba(108,99,255,0.15)' : 'transparent',
            color:       tool === t ? 'var(--accent)' : 'var(--text2)',
            transition: 'all 0.15s',
            textTransform: 'capitalize',
          }}>{t}</button>
        ))}

        {/* Brush size */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, minWidth: '160px' }}>
          <span style={{ fontSize: '12px', color: 'var(--text2)', whiteSpace: 'nowrap' }}>
            Size: {brushSize}px
          </span>
          <input type="range" min={5} max={60} value={brushSize}
            onChange={e => setBrushSize(Number(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--accent)' }}
          />
        </div>

        {/* Clear */}
        <button onClick={clearMask} style={{
          padding: '6px 14px', borderRadius: '6px', fontSize: '13px',
          cursor: 'pointer', border: '1px solid var(--border)',
          background: 'transparent', color: 'var(--error)',
          borderColor: 'rgba(248,113,113,0.3)',
        }}>Clear</button>
      </div>

      {/* Canvas */}
      <div style={{
        position: 'relative', display: 'inline-block',
        border: '1px solid var(--border)', borderRadius: '8px',
        overflow: 'hidden', cursor: tool === 'eraser' ? 'cell' : 'crosshair',
        maxWidth: '100%',
      }}>
        <canvas
          ref={canvasRef}
          style={{ display: 'block', maxWidth: '100%', touchAction: 'none' }}
          onMouseDown={startDraw}
          onMouseMove={draw}
          onMouseUp={stopDraw}
          onMouseLeave={stopDraw}
          onTouchStart={startDraw}
          onTouchMove={draw}
          onTouchEnd={stopDraw}
        />
        {/* Hidden mask canvas */}
        <canvas ref={maskCanvasRef} style={{ display: 'none' }} />
      </div>

      <p style={{ color: 'var(--text2)', fontSize: '12px', marginTop: '8px' }}>
        Paint over the region you want to restore or remove
      </p>
    </div>
  );
}
