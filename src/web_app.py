"""Local dashboard for dataset readiness and benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment_config.yaml"
BENCHMARKS_DIR = ROOT / "results" / "benchmarks"
MODELS = ("tableformer", "tabstructnet", "tabletransformer", "slanet")
DATASETS = ("pubtabnet", "fintabnet", "scitsr", "corporate")
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    return value if isinstance(value, dict) else None


def _safe_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
        raise ValueError("Identificador de execução inválido.")
    candidate = (BENCHMARKS_DIR / run_id).resolve()
    if os.path.commonpath((str(BENCHMARKS_DIR.resolve()), str(candidate))) != str(
        BENCHMARKS_DIR.resolve()
    ):
        raise ValueError("Execução fora do diretório de resultados.")
    return candidate


def _find_runs() -> list[dict[str, Any]]:
    runs = []
    if not BENCHMARKS_DIR.is_dir():
        return runs
    for run_dir in BENCHMARKS_DIR.iterdir():
        report = _read_json(run_dir / "run_summary.json")
        if report is None:
            continue
        report["run_id"] = run_dir.name
        report["modified_at"] = datetime.fromtimestamp(
            (run_dir / "run_summary.json").stat().st_mtime
        ).isoformat(timespec="seconds")
        runs.append(report)
    return sorted(runs, key=lambda item: item["modified_at"], reverse=True)


def _latest_validation() -> dict[str, Any] | None:
    reports_dir = ROOT / "results" / "validation"
    if not reports_dir.is_dir():
        return None
    reports = sorted(
        reports_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in reports:
        try:
            report = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if report is not None:
            report["report_path"] = str(path.relative_to(ROOT))
            return report
    return None


def _dataset_readiness(config: dict[str, Any], run_dir: Path | None) -> list[dict[str, Any]]:
    manifest_counts: dict[str, int] = {}
    if run_dir is not None:
        for row in _read_csv(run_dir / "sample_manifest.csv"):
            dataset = row.get("dataset", "")
            manifest_counts[dataset] = manifest_counts.get(dataset, 0) + 1

    datasets = []
    for name in DATASETS:
        cfg = config.get("datasets", {}).get(name, {})
        raw_dir = ROOT / cfg.get("raw_dir", "")
        if name == "corporate":
            required_paths = [
                raw_dir,
                ROOT / cfg.get("processed_dir", ""),
                ROOT / cfg.get("ground_truth_dir", ""),
                ROOT / cfg.get("table_detection", {}).get("manifest_path", ""),
            ]
        elif name == "scitsr":
            required_paths = [
                raw_dir / "img" / str(cfg.get("split", "test")),
                raw_dir / "structure" / str(cfg.get("split", "test")),
            ]
        else:
            required_paths = [
                raw_dir / str(cfg.get("split", "")),
                raw_dir / cfg.get("annotations_filename", ""),
            ]

        missing = [
            str(path)
            for path in required_paths
            if not path.exists()
        ]
        datasets.append(
            {
                "name": name,
                "enabled": bool(cfg.get("enabled")),
                "split": cfg.get("split") or "curado",
                "configured_sample_size": cfg.get("sample_size"),
                "manifest_samples": manifest_counts.get(name, 0),
                "required_paths_present": not missing,
                "missing_paths": missing,
            }
        )
    return datasets


def _summarize_audit(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = row.get("model", "unknown")
        stats = summary.setdefault(
            model,
            {
                "model": model,
                "workbooks": 0,
                "with_text": 0,
                "text_cells": 0,
                "empty_slots": 0,
                "merged_ranges": 0,
                "bold_cells": 0,
            },
        )
        stats["workbooks"] += 1
        stats["with_text"] += int(int(row.get("nonempty_text_cells") or 0) > 0)
        for key, column in (
            ("text_cells", "nonempty_text_cells"),
            ("empty_slots", "empty_unmerged_grid_slots"),
            ("merged_ranges", "merged_ranges"),
            ("bold_cells", "bold_cells"),
        ):
            stats[key] += int(row.get(column) or 0)
    return [summary[name] for name in sorted(summary)]


def dashboard_payload(run_id: str | None = None) -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    runs = _find_runs()
    selected_run_id = run_id or (runs[0]["run_id"] if runs else None)
    run_dir = _safe_run_dir(selected_run_id) if selected_run_id else None
    report = _read_json(run_dir / "run_summary.json") if run_dir else None
    if report is not None:
        report["run_id"] = selected_run_id

    summary_rows = _read_csv(run_dir / "summary_by_dataset_model.csv") if run_dir else []
    pair_rows = _read_csv(run_dir / "paired_comparisons.csv") if run_dir else []
    audit_rows = _read_csv(run_dir / "xlsx_audit.csv") if run_dir else []
    attribute_rows = _read_csv(run_dir / "attribute_summary.csv") if run_dir else []

    return {
        "project": config.get("experiment", {}).get("name", "TSR benchmark"),
        "runs": runs[:15],
        "report": report,
        "summary": summary_rows,
        "paired_comparisons": pair_rows,
        "xlsx_audit": _summarize_audit(audit_rows),
        "attribute_summary": attribute_rows,
        "dataset_readiness": _dataset_readiness(config, run_dir),
        "validation": _latest_validation(),
        "enabled_models": [
            name
            for name, model_cfg in config.get("models", {}).items()
            if model_cfg.get("enabled")
        ],
        "models": list(MODELS),
    }


def _read_request_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length > 16_384:
        raise ValueError("Solicitação muito grande.")
    try:
        body = handler.rfile.read(length)
        value = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("JSON inválido.") from exc
    if not isinstance(value, dict):
        raise ValueError("O corpo da solicitação deve ser um objeto JSON.")
    return value


def _start_job(kind: str, command: list[str], run_id: str) -> dict[str, Any]:
    with _JOBS_LOCK:
        for job in _JOBS.values():
            process = job.get("process")
            if job["kind"] == kind and process is not None and process.poll() is None:
                raise RuntimeError(f"Já existe uma tarefa {kind} em execução.")

        if kind == "validation":
            log_dir = ROOT / "results" / "validation"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{run_id}.log"
        else:
            run_dir = _safe_run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / f"{kind}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                shell=False,
            )
        job_id = uuid.uuid4().hex
        _JOBS[job_id] = {
            "id": job_id,
            "kind": kind,
            "run_id": run_id,
            "log_path": log_path,
            "process": process,
        }
    return {"id": job_id, "kind": kind, "run_id": run_id, "status": "running"}


def _job_payload(job: dict[str, Any]) -> dict[str, Any]:
    process = job["process"]
    return_code = process.poll()
    log_path = job["log_path"]
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as source:
            lines = source.readlines()[-100:]
    except FileNotFoundError:
        lines = []
    return {
        "id": job["id"],
        "kind": job["kind"],
        "run_id": job["run_id"],
        "status": "running" if return_code is None else (
            "completed" if return_code == 0 else "failed"
        ),
        "return_code": return_code,
        "log_tail": "".join(lines),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TableRecognitionDashboard/1.0"

    def _respond(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _respond_json(self, status: int, value: dict[str, Any]) -> None:
        self._respond(
            status,
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._respond(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/dashboard":
            run_id = parse_qs(parsed.query).get("run_id", [None])[0]
            try:
                self._respond_json(200, dashboard_payload(run_id))
            except (OSError, ValueError, yaml.YAMLError, json.JSONDecodeError) as exc:
                self._respond_json(500, {"error": str(exc)})
            return
        if parsed.path == "/api/jobs":
            with _JOBS_LOCK:
                jobs = [_job_payload(job) for job in _JOBS.values()]
            self._respond_json(200, {"jobs": jobs})
            return
        self._respond_json(404, {"error": "Rota não encontrada."})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = _read_request_json(self)
            if parsed.path == "/api/benchmark":
                models = payload.get("models")
                if not isinstance(models, list) or not models:
                    raise ValueError("Selecione ao menos um modelo.")
                if any(model not in MODELS for model in models):
                    raise ValueError("A lista contém um modelo desconhecido.")
                run_id = payload.get("run_id") or (
                    "ui-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                )
                if not isinstance(run_id, str):
                    raise ValueError("Identificador de execução inválido.")
                run_dir = _safe_run_dir(run_id)
                if run_dir.exists():
                    raise FileExistsError(
                        "Esse identificador já existe; escolha outro para preservar os resultados."
                    )
                command = [
                    sys.executable,
                    "-m",
                    "src.evaluation.benchmark",
                    "--config",
                    str(CONFIG_PATH),
                    "--run-id",
                    run_id,
                    "--models",
                    *models,
                ]
                result = _start_job("benchmark", command, run_id)
            elif parsed.path == "/api/validate":
                run_id = "validation-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                if _safe_run_dir(run_id).exists():
                    run_id += "-" + uuid.uuid4().hex[:6]
                report_path = ROOT / "results" / "validation" / f"{run_id}.json"
                command = [
                    sys.executable,
                    "-m",
                    "src.data_prep.validate_datasets",
                    "--config",
                    str(CONFIG_PATH),
                    "--report",
                    str(report_path),
                ]
                result = _start_job("validation", command, run_id)
            else:
                self._respond_json(404, {"error": "Rota não encontrada."})
                return
        except FileExistsError as exc:
            self._respond_json(409, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._respond_json(409, {"error": str(exc)})
            return
        except (ValueError, OSError) as exc:
            self._respond_json(400, {"error": str(exc)})
            return
        self._respond_json(202, result)

    def log_message(self, format: str, *args: Any) -> None:
        print("[dashboard] " + format % args)


DASHBOARD_HTML = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reconhecimento de Tabelas | Painel</title>
  <style>
    :root{color-scheme:light;--ink:#172433;--muted:#637487;--line:#dce4ec;--blue:#2459a6;--teal:#d9f2ed;--green:#08785f;--amber:#fff0d1;--red:#ffebe8;--bg:#f3f6fa}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "Segoe UI",Arial,sans-serif}
    header{background:#13283c;color:#fff;padding:24px max(22px,calc((100% - 1240px)/2));display:flex;justify-content:space-between;align-items:center;gap:20px}
    header h1{font-size:22px;margin:0}header p{color:#c6d4e2;margin:4px 0 0}
    main{max-width:1240px;margin:24px auto;padding:0 22px 48px}.section{margin-top:22px}.section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px}.section h2{font-size:18px;margin:0}
    .grid{display:grid;gap:14px}.stats{grid-template-columns:repeat(4,minmax(0,1fr))}.datasets{grid-template-columns:repeat(4,minmax(0,1fr))}.models{grid-template-columns:repeat(4,minmax(0,1fr))}
    .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:17px;box-shadow:0 3px 12px #1835520a}.stat-label,.muted{color:var(--muted);font-size:13px}.stat-value{font-size:25px;font-weight:700;margin-top:5px}
    .dataset-title{display:flex;justify-content:space-between;gap:8px;align-items:center;font-weight:700;text-transform:capitalize}.dataset-meta{color:var(--muted);font-size:13px;margin:8px 0}.badge{border-radius:20px;padding:3px 9px;font-size:12px;font-weight:650;white-space:nowrap}.ok{background:var(--teal);color:var(--green)}.warn{background:var(--amber);color:#865600}.bad{background:var(--red);color:#a33225}
    button{border:0;border-radius:8px;background:var(--blue);color:#fff;padding:10px 15px;font-weight:650;cursor:pointer}button.secondary{background:#e8eef5;color:#193451}button:hover{filter:brightness(.95)}button:disabled{opacity:.55;cursor:wait}
    select,input[type=text]{padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink)}
    .run-form{display:flex;flex-wrap:wrap;align-items:center;gap:12px}.model-option{display:flex;align-items:center;gap:7px;background:#f7f9fc;border:1px solid var(--line);border-radius:8px;padding:9px 11px;text-transform:capitalize}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px;background:#fff}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#f7f9fc;color:#536578;font-weight:650}tr:last-child td{border-bottom:0}
    .notice{background:#fff8e9;border:1px solid #f1d89e;border-radius:10px;padding:13px 15px;color:#654c17}.jobs{display:grid;gap:10px}.job{border-left:4px solid var(--blue)}pre{white-space:pre-wrap;max-height:220px;overflow:auto;background:#101b28;color:#d8e2ee;border-radius:8px;padding:12px;font-size:12px}
    .empty{color:var(--muted);padding:20px;text-align:center}.run-picker{display:flex;gap:8px;align-items:center}.footer{color:var(--muted);font-size:12px;margin-top:28px}
    @media(max-width:900px){.stats,.datasets,.models{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:560px){header{align-items:flex-start;flex-direction:column}.stats,.datasets,.models{grid-template-columns:1fr}.section-head{align-items:flex-start;flex-direction:column}.run-form{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body>
  <header><div><h1>Reconhecimento de Tabelas</h1></div></header>
  <main>
    <section class="section">
      <div class="section-head"><h2>Estado da avaliação</h2><div class="run-picker"><label for="runSelect" class="muted">Execução</label><select id="runSelect"></select><button class="secondary" id="refreshBtn">Atualizar</button></div></div>
      <div id="stats" class="grid stats"></div>
      <div id="runNote" style="margin-top:12px"></div>
    </section>
    <section class="section">
      <div class="section-head"><h2>Datasets configurados</h2><button id="validateBtn">Validar dados agora</button></div>
      <div id="datasets" class="grid datasets"></div>
      <p class="muted">A validação verifica imagens e gabaritos. A quantidade no manifesto é da execução selecionada; ausência de caminhos não substitui a validação completa dos arquivos.</p>
    </section>
    <section class="section"><div class="section-head"><h2>Relatório da última validação</h2><span id="validationPath" class="muted"></span></div><div id="validationSummary" class="table-wrap"></div></section>
    <section class="section">
      <div class="section-head"><h2>Iniciar benchmark pareado</h2></div>
      <div class="card">
        <p class="muted">Usa os splits, limites e amostras habilitados em <code>experiment_config.yaml</code>; não altera a configuração. Cada modelo roda no venv configurado.</p>
        <div id="modelChoices" class="run-form models"></div>
        <div class="run-form" style="margin-top:14px"><label>Identificador (opcional) <input id="runId" type="text" placeholder="gerado automaticamente"></label><button id="runBtn">Iniciar nova execução</button></div>
      </div>
    </section>
    <section class="section"><div class="section-head"><h2>Atividade nesta sessão</h2></div><div id="jobs" class="jobs"><div class="empty">Nenhuma tarefa iniciada pelo painel.</div></div></section>
    <section class="section"><div class="section-head"><h2>Métricas por dataset e modelo</h2><span class="muted">média, dispersão e IC bootstrap 95%</span></div><div id="summary" class="table-wrap"></div></section>
    <section class="section"><div class="section-head"><h2>Comparações pareadas</h2><span class="muted">n pequeno: resultados exploratórios</span></div><div id="pairs" class="table-wrap"></div></section>
    <section class="section"><div class="section-head"><h2>Resultados por atributo</h2><div class="run-picker"><span class="muted">Filtrar</span><select id="attributeScope"></select><select id="attributeName"></select></div></div><div id="attributes" class="table-wrap"></div></section>
    <section class="section"><div class="section-head"><h2>Auditoria dos XLSX</h2><span class="muted">texto, vazios, mesclagens e cabeçalhos em negrito</span></div><div id="audit" class="table-wrap"></div></section>
    <p class="footer">O painel não inicia inferências automaticamente. Execute validação e benchmarks somente quando desejar. SLANet requer o ambiente PaddleOCR descrito no guia de configuração.</p>
  </main>
<script>
const $=id=>document.getElementById(id);
const esc=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let selectedRun=null;
let attributeRows=[];
function table(target,rows,columns,empty){if(!rows.length){$(target).innerHTML=`<div class="empty">${esc(empty)}</div>`;return}
  $(target).innerHTML=`<table><thead><tr>${columns.map(c=>`<th>${esc(c[1])}</th>`).join("")}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${esc(r[c[0]])}</td>`).join("")}</tr>`).join("")}</tbody></table>`}
function metric(value,digits=3){return value===""||value==null?"—":Number(value).toFixed(digits)}
function renderAttributes(){
  const scope=$("attributeScope").value;const attribute=$("attributeName").value;
  const rows=attributeRows.filter(r=>(scope==="all"||r.scope===scope)&&(attribute==="all"||r.attribute===attribute));
  table("attributes",rows,[["scope","Escopo"],["attribute","Atributo"],["value","Grupo"],["model","Modelo"],["sample_count","Amostras"],["evaluated_count","Avaliadas"],["mean_teds","TEDS"],["mean_s_teds","S-TEDS"],["mean_cell_f1","Cell-F1"]],"Resumo por atributo indisponível para essa execução.");
}
function render(data){
  const report=data.report;selectedRun=report?.run_id||null;
  const picker=$("runSelect");picker.innerHTML=(data.runs||[]).map(r=>`<option value="${esc(r.run_id)}">${esc(r.run_id)} · ${esc(r.status)}</option>`).join("");
  if(selectedRun)picker.value=selectedRun;
  const counts=report?.sample_counts_by_dataset||{};
  const statuses=report?.model_statuses||{};
  const ready=Object.values(statuses).filter(s=>s==="completed").length;
  $("stats").innerHTML=[
    ["Execução",report?.status||"sem resultados"],
    ["Amostras congeladas",report?.sample_count??"—"],
    ["Datasets",Object.keys(counts).length||"—"],
    ["Runners completos",`${ready}/${Object.keys(statuses).length||data.enabled_models.length||4}`]
  ].map(([label,value])=>`<article class="card"><div class="stat-label">${esc(label)}</div><div class="stat-value">${esc(value)}</div></article>`).join("");
  const statusesText=Object.entries(statuses).map(([m,s])=>`${m}: ${s}`).join(" · ");
  $("runNote").innerHTML=report?`<div class="notice"><b>${esc(report.run_id)}</b> — ${esc(statusesText||"Sem status de runner")}. Código: ${esc(report.code_version||"n/d")}; manifesto: ${esc(report.sample_manifest_sha256||"n/d")}.</div>`:`<div class="notice">Ainda não há um relatório de benchmark concluído disponível. Você pode validar os dados antes de iniciar um.</div>`;
  $("datasets").innerHTML=data.dataset_readiness.map(d=>{
    const pathState=d.required_paths_present?"Caminhos configurados":"Faltam caminhos";
    const badge=d.required_paths_present?"ok":"warn";
    return `<article class="card"><div class="dataset-title"><span>${esc(d.name)}</span><span class="badge ${badge}">${esc(d.enabled?"habilitado":"desabilitado")}</span></div><div class="dataset-meta">split ${esc(d.split)} · amostras configuradas: ${esc(d.configured_sample_size??"todas")} · manifesto da última rodada: ${esc(d.manifest_samples)}</div><span class="badge ${badge}">${esc(pathState)}</span>${d.missing_paths.length?`<p class="muted">Ausentes: ${d.missing_paths.map(esc).join(", ")}</p>`:""}</article>`
  }).join("");
  const validation=data.validation;
  $("validationPath").textContent=validation?.report_path||"Ainda não executada pelo painel";
  if(validation){
    const validationRows=(validation.datasets||[]).map(d=>({
      dataset:d.dataset,
      available:d.available??0,
      validated:d.validated??0,
      loaded:d.loaded??0,
      ignored:d.ignored??0,
      unmatched:`${d.unmatched_images??0} imagens / ${d.unmatched_ground_truth??0} gabaritos`,
      merged:`${(d.samples||[]).filter(s=>s.merged_cells>0).length} amostras`,
      status:d.error||"validado"
    }));
    table("validationSummary",validationRows,[["dataset","Dataset"],["available","Disponíveis"],["validated","Verificadas"],["loaded","Carregadas"],["ignored","Ignoradas"],["unmatched","Sem par"],["merged","Com mesclagens"],["status","Estado"]],"Nenhum resultado de validação.");
  }else{
    $("validationSummary").innerHTML='<div class="empty">Execute a validação para salvar um relatório detalhado de imagens, gabaritos e spans.</div>';
  }
  $("modelChoices").innerHTML=data.enabled_models.map(m=>`<label class="model-option"><input type="checkbox" name="model" value="${esc(m)}" checked>${esc(m)}</label>`).join("");
  table("summary",data.summary,[["dataset","Dataset"],["model","Modelo"],["metric","Métrica"],["n","n"],["mean","Média"],["sample_std","Desvio"],["bootstrap_95_ci_low","IC95% mín."],["bootstrap_95_ci_high","IC95% máx."]],"Sem resumo para essa execução.");
  const pairedRows=data.paired_comparisons.map(row=>({...row,sign_flip_p_two_sided:row.sign_flip_p_two_sided??row.exact_sign_flip_p_two_sided,sign_flip_method:row.sign_flip_method||"exact"}));
  table("pairs",pairedRows,[["dataset","Dataset"],["metric","Métrica"],["model_a","Modelo A"],["model_b","Modelo B"],["n_paired","n pareado"],["mean_paired_difference","Diferença média"],["bootstrap_95_ci_low","IC95% mín."],["bootstrap_95_ci_high","IC95% máx."],["sign_flip_p_two_sided","p teste pareado"],["sign_flip_method","Método"]],"Sem comparações pareadas disponíveis.");
  attributeRows=data.attribute_summary||[];
  const currentScope=$("attributeScope").value||"all";
  const scopes=[...new Set(attributeRows.map(r=>r.scope))];
  $("attributeScope").innerHTML=`<option value="all">Todos os escopos</option>${scopes.map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join("")}`;
  $("attributeScope").value=scopes.includes(currentScope)?currentScope:"all";
  const currentAttribute=$("attributeName").value||"all";
  const attributes=[...new Set(attributeRows.map(r=>r.attribute))];
  $("attributeName").innerHTML=`<option value="all">Todos os atributos</option>${attributes.map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join("")}`;
  $("attributeName").value=attributes.includes(currentAttribute)?currentAttribute:"all";
  renderAttributes();
  table("audit",data.xlsx_audit,[["model","Modelo"],["workbooks","Planilhas"],["with_text","Com texto"],["text_cells","Células textuais"],["empty_slots","Células vazias"],["merged_ranges","Mesclagens"],["bold_cells","Negrito/cabeçalho"]],"Sem auditoria XLSX disponível.");
}
async function loadDashboard(runId){const query=runId?`?run_id=${encodeURIComponent(runId)}`:"";const response=await fetch(`/api/dashboard${query}`);const data=await response.json();if(!response.ok)throw Error(data.error||"Falha ao carregar o painel.");render(data)}
async function loadJobs(){const response=await fetch("/api/jobs");const data=await response.json();$("jobs").innerHTML=data.jobs.length?data.jobs.slice().reverse().map(j=>`<article class="card job"><b>${esc(j.kind==="benchmark"?"Benchmark":"Validação")}: ${esc(j.run_id)}</b> <span class="badge ${j.status==="running"?"warn":j.status==="completed"?"ok":"bad"}">${esc(j.status)}</span>${j.return_code!==null?`<span class="muted">código ${esc(j.return_code)}</span>`:""}<pre>${esc(j.log_tail||"Aguardando saída…")}</pre></article>`).join(""):`<div class="empty">Nenhuma tarefa iniciada pelo painel.</div>`}
async function post(path,payload={}){const response=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw Error(data.error||"A solicitação falhou.");return data}
$("refreshBtn").onclick=()=>loadDashboard($("runSelect").value||null).catch(e=>alert(e.message));
$("runSelect").onchange=()=>loadDashboard($("runSelect").value).catch(e=>alert(e.message));
$("attributeScope").onchange=renderAttributes;$("attributeName").onchange=renderAttributes;
$("validateBtn").onclick=async()=>{try{const j=await post("/api/validate");await loadJobs();alert(`Validação iniciada: ${j.run_id}`)}catch(e){alert(e.message)}};
$("runBtn").onclick=async()=>{const models=[...document.querySelectorAll('input[name="model"]:checked')].map(x=>x.value);try{const j=await post("/api/benchmark",{models,run_id:$("runId").value.trim()||undefined});await loadJobs();alert(`Benchmark iniciado: ${j.run_id}`)}catch(e){alert(e.message)}};
loadDashboard().catch(e=>{$("runNote").innerHTML=`<div class="notice">${esc(e.message)}</div>`});loadJobs();setInterval(loadJobs,3000);
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Painel local do benchmark de tabelas.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host not in ("127.0.0.1", "localhost"):
        raise ValueError("Por segurança, o painel só pode escutar no host local.")
    if not 1 <= args.port <= 65535:
        raise ValueError("A porta precisa estar entre 1 e 65535.")

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(
        f"Painel disponível em http://{args.host}:{args.port} "
        "(Ctrl+C para encerrar)."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando o painel.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
