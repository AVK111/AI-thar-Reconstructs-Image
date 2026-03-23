'use client';
import { useState, useCallback } from 'react';
import MaskCanvas from '@/components/canvas/MaskCanvas';
import CompareSlider from '@/components/results/CompareSlider';
import { inpaintImage, getResultUrl, getInputUrl, type Job, type ModelType } from '@/lib/api';

type Step = 'upload' | 'mask' | 'result';

const MODELS: { id: ModelType; label: string; desc: string; color: string }[] = [
  { id: 'gan',         label: 'GAN',         desc: 'Best quality',   color: '#6c63ff' },
  { id: 'autoencoder', label: 'Autoencoder', desc: 'Fastest',        color: '#4ade80' },
  { id: 'diffusion',   label: 'Diffusion',   desc: 'Most advanced',  color: '#ff6584' },
];

export default function InpaintPage() {
  const [step, setStep]         = useState<Step>('upload');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [maskBlob, setMaskBlob] = useState<Blob | null>(null);
  const [model, setModel]       = useState<ModelType>('gan');
  const [diffSteps, setDiffSteps] = useState(50);
  const [loading, setLoading]   = useState(false);
  const [job, setJob]           = useState<Job | null>(null);
  const [error, setError]       = useState('');
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload a JPG or PNG image.');
      return;
    }
    setImageFile(file);
    setStep('mask');
    setError('');
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, []);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const runInpaint = async () => {
    if (!imageFile || !maskBlob || maskBlob.size === 0) {
      setError('Please draw a mask over the region to restore.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const result = await inpaintImage(imageFile, maskBlob, model, diffSteps);
      setJob(result);
      setStep('result');
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Inpainting failed. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setStep('upload');
    setImageFile(null);
    setMaskBlob(null);
    setJob(null);
    setError('');
  };

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '32px 24px' }}>

      {/* Header */}
      <div style={{ marginBottom: '32px' }}>
        <h1 style={{ fontSize: '32px', fontWeight: 800, marginBottom: '8px' }}>
          Image Inpainting
        </h1>
        <p style={{ color: 'var(--text2)' }}>
          Upload a damaged image, draw over the region to restore, and let AI complete it.
        </p>
      </div>

      {/* Steps indicator */}
      <div style={{ display: 'flex', gap: '0', marginBottom: '32px' }}>
        {(['upload', 'mask', 'result'] as Step[]).map((s, i) => (
          <div key={s} style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
            <div style={{
              width: '28px', height: '28px', borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '12px', fontWeight: 700, flexShrink: 0,
              background: step === s ? 'var(--accent)' :
                          (['upload','mask','result'].indexOf(step) > i) ? 'rgba(108,99,255,0.3)' : 'var(--surface2)',
              color: step === s ? 'white' : 'var(--text2)',
              border: '1px solid',
              borderColor: step === s ? 'var(--accent)' : 'var(--border)',
              transition: 'all 0.2s',
            }}>{i + 1}</div>
            <span style={{
              marginLeft: '8px', fontSize: '13px', fontWeight: 500,
              color: step === s ? 'var(--text)' : 'var(--text2)',
              textTransform: 'capitalize',
            }}>{s}</span>
            {i < 2 && <div style={{ flex: 1, height: '1px', background: 'var(--border)', margin: '0 12px' }} />}
          </div>
        ))}
      </div>

      {error && (
        <div style={{
          padding: '12px 16px', borderRadius: '8px', marginBottom: '20px',
          background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.3)',
          color: 'var(--error)', fontSize: '14px',
        }}>{error}</div>
      )}

      {/* Step 1 — Upload */}
      {step === 'upload' && (
        <div
          onDrop={onDrop}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          style={{
            border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
            borderRadius: '12px', padding: '60px 24px', textAlign: 'center',
            background: dragOver ? 'rgba(108,99,255,0.05)' : 'var(--surface)',
            transition: 'all 0.2s', cursor: 'pointer',
          }}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🖼</div>
          <h3 style={{ fontFamily: 'Syne', fontWeight: 600, marginBottom: '8px' }}>
            Drop your image here
          </h3>
          <p style={{ color: 'var(--text2)', fontSize: '14px', marginBottom: '20px' }}>
            or click to browse — JPG, PNG supported
          </p>
          <button className="btn-primary">Choose Image</button>
          <input id="file-input" type="file" accept="image/*"
            style={{ display: 'none' }} onChange={onFileInput} />
        </div>
      )}

      {/* Step 2 — Draw mask */}
      {step === 'mask' && imageFile && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', gap: '24px', alignItems: 'start' }}>

          {/* Canvas */}
          <div className="card">
            <h3 style={{ fontFamily: 'Syne', fontWeight: 600, marginBottom: '16px' }}>
              Draw over the region to restore
            </h3>
            <MaskCanvas imageFile={imageFile} onMaskReady={setMaskBlob} />
          </div>

          {/* Settings */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>

            {/* Model selection */}
            <div className="card">
              <h4 style={{ fontFamily: 'Syne', fontWeight: 600, fontSize: '14px', marginBottom: '12px' }}>
                Select model
              </h4>
              {MODELS.map(m => (
                <div key={m.id} onClick={() => setModel(m.id)} style={{
                  padding: '10px 12px', borderRadius: '8px', cursor: 'pointer',
                  border: '1px solid', marginBottom: '8px',
                  borderColor: model === m.id ? m.color : 'var(--border)',
                  background: model === m.id ? `${m.color}15` : 'transparent',
                  transition: 'all 0.15s',
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{
                      fontFamily: 'Syne', fontWeight: 600, fontSize: '14px',
                      color: model === m.id ? m.color : 'var(--text)',
                    }}>{m.label}</span>
                    <span style={{
                      fontSize: '11px', color: 'var(--text2)',
                      background: 'var(--surface2)', padding: '2px 6px', borderRadius: '4px',
                    }}>{m.desc}</span>
                  </div>
                </div>
              ))}

              {/* Diffusion steps */}
              {model === 'diffusion' && (
                <div style={{ marginTop: '8px' }}>
                  <p style={{ fontSize: '12px', color: 'var(--text2)', marginBottom: '6px' }}>
                    Inference steps: {diffSteps}
                  </p>
                  <input type="range" min={10} max={100} value={diffSteps}
                    onChange={e => setDiffSteps(Number(e.target.value))}
                    style={{ width: '100%', accentColor: '#ff6584' }}
                  />
                  <p style={{ fontSize: '11px', color: 'var(--text2)', marginTop: '4px' }}>
                    More steps = better quality, slower
                  </p>
                </div>
              )}
            </div>

            {/* Action buttons */}
            <button className="btn-primary" onClick={runInpaint}
              disabled={loading || !maskBlob || maskBlob.size === 0}
              style={{ width: '100%', fontSize: '15px', padding: '12px' }}>
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
                  <span className="spinner" style={{ width: '16px', height: '16px' }} />
                  {model === 'diffusion' ? 'Denoising...' : 'Inpainting...'}
                </span>
              ) : 'Run Inpainting'}
            </button>

            <button className="btn-secondary" onClick={reset} style={{ width: '100%' }}>
              Change Image
            </button>
          </div>
        </div>
      )}

      {/* Step 3 — Result */}
      {step === 'result' && job && (
        <div className="fade-in">
          {/* Metrics bar */}
          <div style={{
            display: 'flex', gap: '12px', marginBottom: '24px', flexWrap: 'wrap',
          }}>
            {[
              { label: 'Model',  val: job.model.toUpperCase(), color: 'var(--accent)' },
              { label: 'PSNR',   val: `${job.psnr} dB`,        color: 'var(--success)' },
              { label: 'SSIM',   val: String(job.ssim),         color: 'var(--success)' },
              { label: 'Time',   val: `${job.time_sec}s`,       color: 'var(--warning)' },
              { label: 'Status', val: job.status,               color: job.status === 'completed' ? 'var(--success)' : 'var(--error)' },
            ].map(s => (
              <div key={s.label} style={{
                background: 'var(--surface)', border: '1px solid var(--border)',
                borderRadius: '8px', padding: '10px 16px',
              }}>
                <div style={{ fontSize: '11px', color: 'var(--text2)', marginBottom: '2px' }}>{s.label}</div>
                <div style={{ fontFamily: 'Syne', fontWeight: 700, color: s.color, fontSize: '15px' }}>{s.val}</div>
              </div>
            ))}
          </div>

          {/* Three panel view */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
            {[
              { label: 'Original Input',    url: job.input_file  ? getInputUrl(job.input_file)  : '' },
              { label: 'AI Restored',       url: job.output_file ? getResultUrl(job.output_file) : '' },
            ].map(panel => (
              <div key={panel.label} className="card" style={{ textAlign: 'center', padding: '12px' }}>
                <p style={{ fontSize: '12px', color: 'var(--text2)', marginBottom: '8px', fontWeight: 500 }}>
                  {panel.label}
                </p>
                {panel.url && (
                  <img src={panel.url} alt={panel.label} style={{
                    width: '100%', borderRadius: '6px', display: 'block',
                  }} />
                )}
              </div>
            ))}

            {/* Before/After slider */}
            <div className="card" style={{ padding: '12px' }}>
              <p style={{ fontSize: '12px', color: 'var(--text2)', marginBottom: '8px', fontWeight: 500, textAlign: 'center' }}>
                Before / After slider
              </p>
              {job.input_file && job.output_file && (
                <CompareSlider
                  beforeUrl={getInputUrl(job.input_file)}
                  afterUrl={getResultUrl(job.output_file)}
                />
              )}
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            {job.output_file && (
              <a href={getResultUrl(job.output_file)} download target="_blank" rel="noreferrer">
                <button className="btn-primary">Download Result</button>
              </a>
            )}
            <button className="btn-secondary" onClick={reset}>Inpaint Another Image</button>
          </div>
        </div>
      )}
    </div>
  );
}
