"use client";

import { useEffect, useState } from "react";

type Problem = { id: string; title: string; topic: string; statement: string; hint: string };
type Dimension = {
  id: string; title: string; score: number; confidence: number;
  probabilities: Record<string, number>; criteria: string[];
  likely_level: number; rubric_feedback: string; revision_tip: string;
};
type Evaluation = { problem_id: string; model: string; total: number; dimensions: Dimension[]; notice: string };

function signed(value: number) {
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

function EvaluationPanel({ result, label }: { result: Evaluation; label?: string }) {
  return <div>
    {label && <h2>{label}</h2>}
    <div className="total"><strong>{result.total.toFixed(2)}<small> / 10</small></strong><span>Rubric total</span></div>
    <p className="muted">{result.model} · probability-weighted scores</p>
    {result.dimensions.map(d => <article className="dimension" key={d.id}>
      <div className="dimension-title"><h3>{d.title}</h3><strong>{d.score.toFixed(2)} / 2</strong></div>
      <progress value={d.score} max={2} aria-label={`${label || "Proof"}: ${d.title} score`} />
      <p>{d.rubric_feedback}</p>
      <details><summary>Rubric levels & uncertainty · confidence {Math.round(d.confidence * 100)}%</summary>
        {d.criteria.map((criterion, i) => <p key={i}><b>Level {i} · {Math.round((d.probabilities[String(i)] || 0) * 100)}%</b><br />{criterion}</p>)}
        <p><b>Revision checklist:</b> {d.revision_tip}</p>
      </details>
    </article>)}
  </div>;
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(typeof data?.detail === "string" ? data.detail : `Request failed (${response.status}). Please try again.`);
  }
  return response.json();
}

export default function Home() {
  const [problems, setProblems] = useState<Problem[]>([]);
  const [selected, setSelected] = useState("__new__");
  const [saved, setSaved] = useState<Problem[]>([]);
  const [customStatement, setCustomStatement] = useState("");
  const [customTitle, setCustomTitle] = useState("");
  const [saveNotice, setSaveNotice] = useState("");
  const [proof, setProof] = useState("");
  const [comparisonMode, setComparisonMode] = useState(false);
  const [secondProof, setSecondProof] = useState("");
  const [result, setResult] = useState<{ a: Evaluation; b?: Evaluation } | null>(null);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [hint, setHint] = useState(false);
  const allProblems = [...problems, ...saved];
  const problem = allProblems.find(p => p.id === selected);
  const isNew = selected === "__new__";
  const isSaved = saved.some(p => p.id === selected);
  const statement = isNew ? customStatement : problem?.statement || "";

  async function load() {
    setLoading(true); setError("");
    try {
      const data = await api<Problem[]>("/api/problems");
      setProblems(data); setSelected(data[0]?.id || "__new__");
    } catch (e) { setError(e instanceof Error ? e.message : "Unable to load problems."); }
    finally { setLoading(false); }
  }
  useEffect(() => {
    void load();
    try {
      const stored: unknown = JSON.parse(localStorage.getItem("proof-practice-questions") || "[]");
      if (Array.isArray(stored)) {
        setSaved(stored.filter((p): p is Problem =>
          p && typeof p.id === "string" && p.id.startsWith("saved-") &&
          typeof p.title === "string" && p.title.length <= 100 &&
          typeof p.statement === "string" && p.statement.trim().length > 0 &&
          p.statement.length <= 2000
        ).map(p => ({ ...p, topic: "Saved question", hint: "" })));
      }
    } catch { setSaveNotice("Saved questions could not be loaded from this browser."); }
  }, []);

  function saveQuestion() {
    if (!statement.trim()) return;
    const question: Problem = {
      id: `saved-${crypto.randomUUID()}`,
      title: customTitle.trim() || statement.trim().slice(0, 70),
      statement: statement.trim(), topic: "Saved question", hint: "",
    };
    const updated = [...saved, question];
    try {
      localStorage.setItem("proof-practice-questions", JSON.stringify(updated));
      setSaved(updated); setSelected(question.id); setResult(null);
      setSaveNotice("Question saved in this browser.");
    } catch { setSaveNotice("Unable to save. Browser storage may be disabled or full."); }
  }

  function removeQuestion() {
    const updated = saved.filter(p => p.id !== selected);
    try {
      localStorage.setItem("proof-practice-questions", JSON.stringify(updated));
      setSaved(updated); setSelected("__new__"); setResult(null); setProof("");
      setSaveNotice("Saved question removed.");
    } catch { setSaveNotice("Unable to remove the saved question from browser storage."); }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault(); setBusy(true); setError(""); setResult(null);
    try {
      const question = isNew || isSaved
        ? { custom_statement: statement } : { problem_id: selected };
      const init: RequestInit = {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...question, proof, ...(comparisonMode ? { second_proof: secondProof } : {}) }),
        signal: AbortSignal.timeout(65000),
      };
      if (comparisonMode) {
        const data = await api<{ answer_a: Evaluation; answer_b: Evaluation }>("/api/compare", init);
        setResult({ a: data.answer_a, b: data.answer_b });
      } else {
        setResult({ a: await api<Evaluation>("/api/evaluate", init) });
      }
    } catch (e) { setError(e instanceof Error ? e.message : "Unable to grade your proof."); }
    finally { setBusy(false); }
  }

  return <main>
    <header><span className="brand">∴ PROOF PRACTICE</span><span className="badge">Powered by Jev</span></header>
    {loading && <p role="status">Loading questions…</p>}
    {error && <div className="error" role="alert">{error} {!problems.length && <button onClick={() => void load()}>Retry</button>}</div>}
    <div className={`workspace ${comparisonMode ? "comparison-workspace" : ""}`}>
      <section className="card editor">
        <label htmlFor="problem">01 / Choose your question</label>
        <select id="problem" value={selected} disabled={busy} onChange={e => {
          setSelected(e.target.value); setProof(""); setSecondProof(""); setResult(null); setHint(false); setError(""); setSaveNotice("");
        }}>
          <option value="__new__">Write your own question</option>
          {allProblems.map(p => <option key={p.id} value={p.id}>{p.title} · {p.topic}</option>)}
        </select>
        {isNew ? <div className="custom-question">
          <label htmlFor="question-title">Title (optional)</label>
          <input id="question-title" value={customTitle} maxLength={100} disabled={busy}
            placeholder="e.g. Divisibility by three" onChange={e => setCustomTitle(e.target.value)} />
          <label htmlFor="question-statement">Question or claim to prove</label>
          <textarea id="question-statement" className="question-input" value={customStatement}
            maxLength={2000} disabled={busy} placeholder="Write the claim, assumptions, and variable domains."
            onChange={e => { setCustomStatement(e.target.value); setResult(null); setSaveNotice(""); }} />
          <small>{customStatement.length} / 2,000 characters</small>
          <div className="question-actions"><button disabled={busy || !customStatement.trim()} onClick={saveQuestion}>Save question</button></div>
          <p className="muted">Saving is optional. Saved questions stay in this browser only; proofs are not saved.</p>
        </div> : problem && <>
          <span className="topic">{problem.topic}</span>
          <h2>{problem.statement}</h2>
          {problem.hint && <><button className="link" onClick={() => setHint(!hint)} aria-expanded={hint}>{hint ? "Hide hint" : "Need a starting point?"}</button>
            {hint && <p className="hint">{problem.hint}</p>}</>}
          {isSaved && <div className="question-actions">
            <button disabled={busy} onClick={() => {
              setCustomStatement(problem.statement); setCustomTitle(problem.title);
              setSelected("__new__"); setResult(null); setSaveNotice("");
            }}>Edit a copy</button>
            <button disabled={busy} onClick={removeQuestion}>Remove saved question</button>
          </div>}
        </>}
        {saveNotice && <p className="muted" role="status">{saveNotice}</p>}
        {(isNew || isSaved) && <p className="muted">Custom questions are evaluated without a reference proof. Feedback may be less reliable for advanced or ambiguous claims.</p>}
        <form onSubmit={submit}>
          <label className="compare-toggle"><input type="checkbox" checked={comparisonMode} disabled={busy}
            onChange={e => { setComparisonMode(e.target.checked); setResult(null); }} /> Compare two answers to this question</label>
          <div className={comparisonMode ? "answer-grid" : ""}>
            <div><label htmlFor="proof">{comparisonMode ? "Answer A" : "02 / Build your argument"}</label>
              <textarea id="proof" value={proof} maxLength={8000} required disabled={busy}
                placeholder="Let… Assume… Therefore…" onChange={e => { setProof(e.target.value); setResult(null); }} />
              <small>{proof.length.toLocaleString()} / 8,000 characters</small>
            </div>
            {comparisonMode && <div><label htmlFor="second-proof">Answer B</label>
              <textarea id="second-proof" value={secondProof} maxLength={8000} required disabled={busy}
                placeholder="Write an alternative proof or revised answer." onChange={e => { setSecondProof(e.target.value); setResult(null); }} />
              <small>{secondProof.length.toLocaleString()} / 8,000 characters</small>
            </div>}
          </div>
          <div className="actions">
            <button className="primary" disabled={busy || !proof.trim() || !statement.trim() || (comparisonMode && !secondProof.trim())}>{busy ? "Jev is evaluating…" : comparisonMode ? "Compare answers" : "Evaluate proof →"}</button></div>
          <small className="privacy">{comparisonMode ? "Both answers are evaluated independently (two Jev requests)." : "Your proof is sent to TypeSafe for evaluation."} This app does not save submissions.</small>
        </form>
      </section>
      <section className="card feedback" aria-live="polite" aria-busy={busy}>
        <p className="eyebrow">03 / REFLECT & REFINE</p>
        {!result ? <><h2>{busy ? "Evaluating your proof…" : "Proof feedback"}</h2>
          <p>Each dimension is scored from 0 to 2. The equally weighted total is out of 10, not a correctness verdict.</p>
          <ol className="rubric-list"><li>Setup and assumptions</li><li>Logical reasoning</li><li>Justification</li><li>Clarity and notation</li><li>Completeness and conclusion</li></ol>
          <p className="muted">AI feedback can be wrong. Use it to guide revision, not to certify a proof.</p></> : <>
          {result.b && <>
            <h2>Answer comparison</h2>
            <p className="muted">Differences are B minus A. Higher scores are rubric judgments, not proof verification. Small differences may reflect model variability.</p>
            <div className="comparison-table"><table><thead><tr><th>Dimension</th><th>A</th><th>B</th><th>B − A</th></tr></thead>
              <tbody>{result.a.dimensions.map(a => {
                const b = result.b!.dimensions.find(d => d.id === a.id)!;
                return <tr key={a.id}><th>{a.title}</th><td>{a.score.toFixed(2)}</td><td>{b.score.toFixed(2)}</td><td>{signed(b.score - a.score)}</td></tr>;
              })}<tr><th>Total / 10</th><td>{result.a.total.toFixed(2)}</td><td>{result.b.total.toFixed(2)}</td><td>{signed(result.b.total - result.a.total)}</td></tr></tbody>
            </table></div>
          </>}
          <div className={result.b ? "answer-grid" : ""}>
            <EvaluationPanel result={result.a} label={result.b ? "Answer A" : undefined} />
            {result.b && <EvaluationPanel result={result.b} label="Answer B" />}
          </div>
          <p className="muted notice">{result.a.notice}</p>
        </>}
      </section>
    </div>
  </main>;
}
