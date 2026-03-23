import Link from 'next/link';

export default function Home() {
  const models = [
    { name: 'GAN', desc: 'Best quality results. Uses adversarial training for photorealistic reconstruction.', psnr: '26.57', ssim: '0.9133', time: '0.69s', color: '#6c63ff' },
    { name: 'Autoencoder', desc: 'Fastest inference. Compresses image to latent space and reconstructs.', psnr: '23.49', ssim: '0.8874', time: '0.06s', color: '#4ade80' },
    { name: 'Diffusion', desc: 'Most advanced. Iteratively denoises to reconstruct masked regions.', psnr: '14.63', ssim: '0.7684', time: '9.76s', color: '#ff6584' },
  ];

  const steps = [
    { n: '01', title: 'Upload image', desc: 'Upload any damaged or complete image in JPG or PNG format.' },
    { n: '02', title: 'Draw mask', desc: 'Paint over the damaged or unwanted region using the brush tool.' },
    { n: '03', title: 'Choose model', desc: 'Select GAN for best quality, Autoencoder for speed, or Diffusion for state-of-the-art.' },
    { n: '04', title: 'Get result', desc: 'AI reconstructs the masked region. Compare before and after instantly.' },
  ];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 24px' }}>

      {/* Hero */}
      <section style={{ textAlign: 'center', padding: '80px 0 60px' }}>
        <div style={{
          display: 'inline-block', padding: '4px 14px', borderRadius: '99px',
          background: 'rgba(108,99,255,0.12)', border: '1px solid rgba(108,99,255,0.3)',
          color: 'var(--accent)', fontSize: '13px', fontWeight: 500, marginBottom: '24px',
        }}>
          AI-Powered Image Inpainting
        </div>
        <h1 style={{ fontSize: 'clamp(36px, 6vw, 64px)', fontWeight: 800, lineHeight: 1.1, marginBottom: '20px' }}>
          Restore Images with<br />
          <span style={{ color: 'var(--accent)' }}>Deep Learning</span>
        </h1>
        <p style={{ color: 'var(--text2)', fontSize: '18px', maxWidth: '560px', margin: '0 auto 36px', lineHeight: 1.7 }}>
          Upload a damaged photo, draw over the region to fix, and let our AI models reconstruct it realistically.
        </p>
        <div style={{ display: 'flex', gap: '12px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <Link href="/inpaint">
            <button className="btn-primary" style={{ fontSize: '16px', padding: '12px 32px' }}>
              Start Inpainting
            </button>
          </Link>
          <Link href="/history">
            <button className="btn-secondary" style={{ fontSize: '16px', padding: '12px 32px' }}>
              View History
            </button>
          </Link>
        </div>
      </section>

      {/* How it works */}
      <section style={{ padding: '40px 0' }}>
        <h2 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '32px', textAlign: 'center' }}>
          How it works
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px' }}>
          {steps.map(s => (
            <div key={s.n} className="card" style={{ position: 'relative', overflow: 'hidden' }}>
              <div style={{
                fontSize: '48px', fontFamily: 'Syne', fontWeight: 800,
                color: 'rgba(108,99,255,0.08)', position: 'absolute', top: '8px', right: '16px',
                lineHeight: 1,
              }}>{s.n}</div>
              <div style={{
                width: '32px', height: '32px', borderRadius: '8px',
                background: 'rgba(108,99,255,0.15)', border: '1px solid rgba(108,99,255,0.3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: 'var(--accent)', fontFamily: 'Syne', fontWeight: 700, fontSize: '13px',
                marginBottom: '12px',
              }}>{s.n}</div>
              <h3 style={{ fontFamily: 'Syne', fontWeight: 600, fontSize: '16px', marginBottom: '8px' }}>{s.title}</h3>
              <p style={{ color: 'var(--text2)', fontSize: '14px', lineHeight: 1.6 }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Models */}
      <section style={{ padding: '40px 0 80px' }}>
        <h2 style={{ fontSize: '28px', fontWeight: 700, marginBottom: '8px', textAlign: 'center' }}>
          Three models, one platform
        </h2>
        <p style={{ color: 'var(--text2)', textAlign: 'center', marginBottom: '32px' }}>
          Evaluated on 20 test images from CelebA and ImageNet datasets
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          {models.map(m => (
            <div key={m.name} className="card" style={{ borderColor: `${m.color}30` }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px',
              }}>
                <div style={{
                  width: '10px', height: '10px', borderRadius: '50%', background: m.color,
                }} />
                <h3 style={{ fontFamily: 'Syne', fontWeight: 700, fontSize: '18px' }}>{m.name}</h3>
              </div>
              <p style={{ color: 'var(--text2)', fontSize: '14px', marginBottom: '16px', lineHeight: 1.6 }}>{m.desc}</p>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {[
                  { label: 'PSNR', val: `${m.psnr} dB` },
                  { label: 'SSIM', val: m.ssim },
                  { label: 'Time', val: m.time },
                ].map(stat => (
                  <div key={stat.label} style={{
                    background: 'var(--surface2)', borderRadius: '6px',
                    padding: '6px 12px', fontSize: '12px',
                  }}>
                    <span style={{ color: 'var(--text2)' }}>{stat.label}: </span>
                    <span style={{ color: m.color, fontWeight: 600 }}>{stat.val}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
