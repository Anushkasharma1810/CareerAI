import React, { useState, useCallback } from 'react';
import './App.css';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

const API = 'http://localhost:5000';

// ── Score Ring ───────────────────────────────────────────────────────────────
function ScoreRing({ score }) {
  const r = 65;
  const circ = 2 * Math.PI * r;
  const color = score >= 70 ? '#51cf66' : score >= 45 ? '#ffd43b' : '#ff6b6b';
  const offset = circ - (score / 100) * circ;
  return (
    <div className="score-ring">
      <svg viewBox="0 0 160 160">
        <circle className="score-ring-bg" cx="80" cy="80" r={r} />
        <circle
          className="score-ring-fill"
          cx="80" cy="80" r={r}
          stroke={color}
          strokeDasharray={circ}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="score-text">
        <div className="score-number" style={{ color }}>{score}</div>
        <div className="score-label">/ 100</div>
      </div>
    </div>
  );
}

// ── ProgressBar ──────────────────────────────────────────────────────────────
function ProgressBar({ value, color = '#6c63ff', height = 5 }) {
  return (
    <div className="progress-bar" style={{ height }}>
      <div
        className="progress-fill"
        style={{ width: `${Math.min(100, value)}%`, background: color }}
      />
    </div>
  );
}

// ── Chip list ────────────────────────────────────────────────────────────────
function ChipList({ items, variant = 'neutral' }) {
  if (!items || items.length === 0)
    return <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>None detected</span>;
  return (
    <div className="chip-list">
      {items.map(s => (
        <span key={s} className={`chip chip-${variant}`}>{s}</span>
      ))}
    </div>
  );
}

// ── Home Page (upload + analyze) ─────────────────────────────────────────────
function HomePage({ onResult }) {
  const [file, setFile] = useState(null);
  const [jdText, setJdText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onDrop = useCallback(accepted => {
    if (accepted.length > 0) setFile(accepted[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

  const handleAnalyze = async () => {
    setError('');
    if (!file) { setError('Please upload a PDF resume.'); return; }
    if (!jdText.trim()) { setError('Please enter a job description.'); return; }

    setLoading(true);
    try {
      const fd = new FormData();
      fd.append('resume', file);
      fd.append('jd_text', jdText);
      const res = await axios.post(`${API}/analyze`, fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      });
      onResult(res.data);
    } catch (e) {
      const msg = e.response?.data?.error || e.message || 'Analysis failed.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="spinner" />
        <p style={{ color: 'var(--text-muted)' }}>Analyzing your resume…</p>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.5rem' }}>
          Running ML inference and matching engine
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="hero">
        <h1>CareerAI</h1>
        <p>Upload your resume, paste a job description, and get an AI-powered match analysis powered by a real trained ML model — no OpenAI required.</p>
      </div>

      {error && (
        <div className="alert alert-error">
          <span>⚠</span>
          <span>{error}</span>
        </div>
      )}

      <div className="input-grid">
        <div className="card">
          <div className="card-title">📄 Resume (PDF)</div>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''} ${file ? 'has-file' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone-icon">{file ? '✅' : '📂'}</div>
            <div className="dropzone-title">
              {file ? file.name : 'Drop your PDF resume here'}
            </div>
            <div className="dropzone-sub">
              {file
                ? `${(file.size / 1024).toFixed(0)} KB — click to change`
                : 'or click to browse · PDF only · max 10 MB'}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-title">💼 Job Description</div>
          <textarea
            className="textarea-jd"
            placeholder="Paste the job description here…

Example:
We are looking for a Senior ML Engineer.
Required: Python, TensorFlow, Docker, AWS, Kubernetes, SQL
Preferred: PyTorch, MLflow, Spark"
            value={jdText}
            onChange={e => setJdText(e.target.value)}
          />
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
            {jdText.length} characters
          </div>
        </div>
      </div>

      <div className="analyze-bar">
        <button
          className="btn-primary"
          onClick={handleAnalyze}
          disabled={!file || !jdText.trim()}
        >
          <span>⚡</span> Analyze Resume
        </button>
      </div>
    </>
  );
}

// ── Results Page ─────────────────────────────────────────────────────────────
function ResultsPage({ data, onBack }) {
  const match = data.match_analysis || {};
  const pred = data.role_prediction || {};
  const resume = data.resume || {};
  const jd = data.job_description || {};

  const score = match.overall_score ?? 0;

  const allProbs = pred.all_probabilities
    ? Object.entries(pred.all_probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <>
      <div className="results-header">
        <div className="section-title">📊 Analysis Results</div>
        <button className="btn-secondary" onClick={onBack}>← New Analysis</button>
      </div>

      {/* Row 1: Score + Role + Skills */}
      <div className="results-grid">
        {/* Score */}
        <div className="card">
          <div className="card-title">🎯 Match Score</div>
          <div className="score-container">
            <ScoreRing score={score} />
            <div className="score-breakdown">
              <div className="score-breakdown-item">
                <div className="score-breakdown-label">
                  <span>Skill Overlap (60%)</span>
                  <span>{match.skill_score ?? 0}%</span>
                </div>
                <ProgressBar value={match.skill_score ?? 0} color="#51cf66" />
              </div>
              <div className="score-breakdown-item">
                <div className="score-breakdown-label">
                  <span>Semantic Similarity (25%)</span>
                  <span>{match.cosine_score ?? 0}%</span>
                </div>
                <ProgressBar value={match.cosine_score ?? 0} color="#6c63ff" />
              </div>
              <div className="score-breakdown-item">
                <div className="score-breakdown-label">
                  <span>Keyword Density (15%)</span>
                  <span>{match.keyword_score ?? 0}%</span>
                </div>
                <ProgressBar value={match.keyword_score ?? 0} color="#4ecdc4" />
              </div>
            </div>
            <div className="disclaimer" style={{ marginTop: '1rem', textAlign: 'left' }}>
              {match.methodology}
            </div>
          </div>
        </div>

        {/* Right column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Role Prediction */}
          <div className="card">
            <div className="card-title">🤖 Predicted Job Role</div>
            <div className="role-chip">
              <span>✦</span> {pred.predicted_role || 'Unknown'}
            </div>
            {pred.confidence != null && (
              <div className="confidence-bar-wrap">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '4px' }}>
                  <span>Model Confidence</span>
                  <span>{(pred.confidence * 100).toFixed(1)}%</span>
                </div>
                <ProgressBar value={pred.confidence * 100} color="#a78bfa" height={6} />
              </div>
            )}
            {pred.note && <div className="disclaimer" style={{ marginTop: '0.75rem' }}>{pred.note}</div>}

            {allProbs.length > 0 && (
              <>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '1rem', marginBottom: '0.5rem' }}>All role probabilities</div>
                <div className="prob-list">
                  {allProbs.map(([role, prob]) => (
                    <div className="prob-item" key={role}>
                      <span className="prob-item-name">{role}</span>
                      <ProgressBar value={prob * 100} color={role === pred.predicted_role ? '#a78bfa' : '#2d3047'} />
                      <span className="prob-item-val">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem' }}>
            {[
              { label: 'Resume Pages', value: resume.page_count ?? '—' },
              { label: 'JD Words', value: jd.word_count ?? '—' },
              { label: 'Matched Skills', value: match.matched_skills?.length ?? 0 },
            ].map(s => (
              <div className="card" key={s.label} style={{ textAlign: 'center', padding: '1rem' }}>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--accent)' }}>{s.value}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Row 2: Skills */}
      <div className="results-row">
        <div className="card">
          <div className="card-title">✅ Matched Skills</div>
          <ChipList items={match.matched_skills} variant="matched" />
        </div>
        <div className="card">
          <div className="card-title">⚠ Missing Skills</div>
          <ChipList items={match.missing_skills} variant="missing" />
        </div>
        <div className="card">
          <div className="card-title">📋 Resume Skills</div>
          <ChipList items={resume.skills?.slice(0, 20)} variant="neutral" />
        </div>
      </div>

      {/* Row 3: Recommendations + Info */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.5rem' }}>
        <div className="card">
          <div className="card-title">💡 Recommendations</div>
          <div className="rec-list">
            {(match.recommendations || []).map((r, i) => (
              <div className="rec-item" key={i}>
  <span className="rec-icon">→</span>
  <span>
    {r.split(/(https?:\/\/[^\s]+|(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s]*)?)/g).map((part, idx) => {
      if (/^(https?:\/\/|www\.)/.test(part) || /^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/.test(part)) {
        const url = part.startsWith('http') ? part : `https://${part}`;
        return (
          <a
            key={idx}
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'var(--accent)', textDecoration: 'underline' }}
          >
            {part}
          </a>
        );
      }
      return <React.Fragment key={idx}>{part}</React.Fragment>;
    })}
  </span>
</div>
            ))}
            {(!match.recommendations || match.recommendations.length === 0) && (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>No recommendations generated.</p>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-title">🔑 JD Keywords Matched</div>
          <ChipList items={match.keywords_matched?.slice(0, 20)} variant="neutral" />
          <div style={{ marginTop: '1rem' }}>
            <div className="card-title" style={{ marginTop: '1rem' }}>💼 Detected JD Role</div>
            <div style={{ color: jd.job_title ? 'var(--text)' : 'var(--text-muted)', fontSize: '0.9rem' }}>
              {jd.job_title || 'Could not auto-detect role'}
            </div>
          </div>
          {resume.warnings?.length > 0 && (
            <div className="alert alert-warning" style={{ marginTop: '1rem' }}>
              <span>⚠</span>
              <span>{resume.warnings.join(' ')}</span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ── Model Evaluation Page ─────────────────────────────────────────────────────
function EvalPage() {
  const [info, setInfo] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState('');

  React.useEffect(() => {
    axios.get(`${API}/model-info`)
      .then(r => setInfo(r.data.data))
      .catch(e => setError(e.response?.data?.error || 'Could not load model info'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-overlay"><div className="spinner" /></div>;
  if (error) return <div className="alert alert-error"><span>⚠</span><span>{error}</span></div>;
  if (!info) return null;

  const bestName = info.best_model;
  const models = info.models || {};
  const bestKey = bestName === 'Logistic Regression' ? 'logistic_regression' : 'random_forest';
  const bestMetrics = models[bestKey] || {};
  const labels = info.label_classes || [];

  return (
    <>
      <div className="section-title">📈 Model Evaluation</div>

      {/* ── Selection methodology banner ── */}
      <div className="card" style={{ marginBottom: '1.5rem', borderColor: 'var(--accent)' }}>
        <div className="card-title">⚙️ Model Selection Methodology</div>
        <div style={{ fontSize: '0.875rem', lineHeight: 1.7 }}>
          <strong>Selection criterion:</strong> {info.selection_criterion}
          <br />
          <strong>Selected:</strong> {bestName}
          <br />
          <strong>Reason:</strong> {info.selection_reason}
        </div>
        <div className="disclaimer" style={{ marginTop: '0.75rem' }}>
          The test set was locked before CV ran and was only evaluated ONCE after selection.
          CV scores (training set) were the sole basis for picking the model.
          Test-set metrics below are the final reported results.
        </div>
      </div>

      {/* ── Overview pills ── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="card-title">🏆 Best Model: {bestName}</div>
        <div className="info-row">
          <span className="info-pill">TF-IDF Vectorizer</span>
          <span className="info-pill">{info.tfidf_features?.toLocaleString()} features</span>
          <span className="info-pill">v{info.version}</span>
          <span className="info-pill">{labels.length} job categories</span>
          <span className="info-pill">{info.cv_folds}-Fold CV</span>
        </div>
      </div>

      {/* ── Final test-set metrics (winner only) ── */}
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600 }}>
        Final Held-Out Test-Set Metrics — {bestName}
      </div>
      <div className="metrics-grid">
        {[
          { name: 'Accuracy',  val: bestMetrics.accuracy,           color: '#51cf66' },
          { name: 'Precision', val: bestMetrics.precision_weighted,  color: '#6c63ff' },
          { name: 'Recall',    val: bestMetrics.recall_weighted,     color: '#4ecdc4' },
          { name: 'F1-Score',  val: bestMetrics.f1_weighted,         color: '#a78bfa' },
        ].map(m => (
          <div className="metric-card" key={m.name}>
            <div className="metric-value" style={{ color: m.color }}>
              {m.val != null ? (m.val * 100).toFixed(1) + '%' : '—'}
            </div>
            <div className="metric-name">{m.name}</div>
          </div>
        ))}
      </div>

      {/* ── Model comparison table ── */}
      <div className="section-title" style={{ marginTop: '1.5rem', fontSize: '1.1rem' }}>
        Model Comparison
      </div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.75rem' }}>
        CV F1 = training-set cross-validation (used for selection) · Test F1 = held-out test set (reported after selection)
      </div>
      <div className="model-compare-grid">
        {[
          { key: 'logistic_regression', name: 'Logistic Regression' },
          { key: 'random_forest',       name: 'Random Forest' },
        ].map(m => {
          const mt = models[m.key] || {};
          const isBest = m.name === bestName;
          return (
            <div className={`model-card ${isBest ? 'best' : ''}`} key={m.key}>
              {isBest && <div className="model-badge">★ Selected by CV</div>}
              <div style={{ fontWeight: 600, marginBottom: '0.75rem' }}>{m.name}</div>

              <div style={{ fontSize: '0.75rem', color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem' }}>
                Training CV (selection basis)
              </div>
              <div className="model-stats" style={{ marginBottom: '0.75rem' }}>
                {[
                  ['CV F1 Mean', mt.cv_f1_mean],
                  ['CV F1 Std ±', mt.cv_f1_std],
                ].map(([label, val]) => (
                  <div className="stat-row" key={label}>
                    <span className="stat-label">{label}</span>
                    <span className="stat-value" style={{ color: isBest ? 'var(--accent)' : undefined }}>
                      {val != null ? (val * 100).toFixed(2) + '%' : '—'}
                    </span>
                  </div>
                ))}
                {mt.cv_fold_scores && (
                  <div className="stat-row" style={{ flexDirection: 'column', gap: '2px' }}>
                    <span className="stat-label">Per-fold F1</span>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      {mt.cv_fold_scores.map(s => (s * 100).toFixed(1) + '%').join(' · ')}
                    </span>
                  </div>
                )}
              </div>

              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: '0.4rem' }}>
                Held-out test set
              </div>
              <div className="model-stats">
                {[
                  ['Accuracy',       mt.accuracy],
                  ['F1-Score (wtd)', mt.f1_weighted],
                  ['Precision (wtd)',mt.precision_weighted],
                  ['Recall (wtd)',   mt.recall_weighted],
                ].map(([label, val]) => (
                  <div className="stat-row" key={label}>
                    <span className="stat-label">{label}</span>
                    <span className="stat-value">
                      {val != null ? (val * 100).toFixed(2) + '%' : '—'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Confusion Matrix ── */}
      {bestMetrics.confusion_matrix && (
        <div className="card" style={{ marginTop: '1.5rem' }}>
          <div className="card-title">🔢 Confusion Matrix — {bestName} (test set)</div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.5rem' }}>
            Rows = Actual class, Columns = Predicted class. Diagonal = correct predictions.
          </p>
          <ConfusionMatrix matrix={bestMetrics.confusion_matrix} labels={labels} />
        </div>
      )}

      {/* ── Dataset Info ── */}
      <div className="card" style={{ marginTop: '1.5rem' }}>
        <div className="card-title">📦 Dataset Information</div>
        <div className="model-stats">
          {info.dataset && Object.entries(info.dataset).map(([k, v]) => (
            typeof v !== 'object' && (
              <div className="stat-row" key={k}>
                <span className="stat-label">{k.replace(/_/g, ' ')}</span>
                <span className="stat-value">{String(v)}</span>
              </div>
            )
          ))}
        </div>
        <div className="disclaimer" style={{ marginTop: '1rem' }}>
          {info.dataset?.note}
        </div>
      </div>
    </>
  );
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix || matrix.length === 0) return null;
  const maxVal = Math.max(...matrix.flat());
  const shortLabel = l => l.split(' ').map(w => w[0]).join('');

  return (
    <div className="cm-container">
      <table className="cm-table">
        <thead>
          <tr>
            <th>A\P</th>
            {labels.map(l => <th key={l} title={l}>{shortLabel(l)}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th title={labels[i]}>{shortLabel(labels[i])}</th>
              {row.map((v, j) => {
                const pct = maxVal > 0 ? v / maxVal : 0;
                const cls = pct > 0.6 ? 'cm-cell-high' : pct > 0.2 ? 'cm-cell-mid' : 'cm-cell-low';
                return <td key={j} className={cls}>{v}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
        Labels: {labels.map((l, i) => `${shortLabel(l)}=${l}`).join(' · ')}
      </div>
    </div>
  );
}

// ── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState('home');
  const [results, setResults] = useState(null);

  const handleResult = data => {
    setResults(data);
    setTab('results');
  };

  const handleBack = () => {
    setResults(null);
    setTab('home');
  };

  return (
    <div className="app-wrapper">
      <header className="header">
        <div className="header-inner">
          <div className="logo" onClick={() => { setTab('home'); setResults(null); }}>
            <div className="logo-icon">⚡</div>
            CareerAI
          </div>
          <nav className="nav-tabs">
            <button
              className={`nav-tab ${tab === 'home' || tab === 'results' ? 'active' : ''}`}
              onClick={() => { if (results) setTab('results'); else setTab('home'); }}
            >
              {results ? 'Results' : 'Analyze'}
            </button>
            <button
              className={`nav-tab ${tab === 'eval' ? 'active' : ''}`}
              onClick={() => setTab('eval')}
            >
              Model Evaluation
            </button>
          </nav>
        </div>
      </header>

      <main className="main">
        {tab === 'home' && <HomePage onResult={handleResult} />}
        {tab === 'results' && results && <ResultsPage data={results} onBack={handleBack} />}
        {tab === 'results' && !results && <HomePage onResult={handleResult} />}
        {tab === 'eval' && <EvalPage />}
      </main>
    </div>
  );
}
