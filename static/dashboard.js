const state = {
  jobs: [],
  selectedJobId: null,
  selectedTranscript: "",
  pollTimer: null,
};

const els = {
  form: document.querySelector("#uploadForm"),
  chooseButton: document.querySelector("#chooseButton"),
  files: document.querySelector("#audioFiles"),
  dropzone: document.querySelector("#dropzone"),
  fileList: document.querySelector("#fileList"),
  model: document.querySelector("#model"),
  language: document.querySelector("#language"),
  task: document.querySelector("#task"),
  chunkSeconds: document.querySelector("#chunkSeconds"),
  speakerMode: document.querySelector("#speakerMode"),
  initialPrompt: document.querySelector("#initialPrompt"),
  includeTimestamps: document.querySelector("#includeTimestamps"),
  uploadMeter: document.querySelector("#uploadMeter"),
  uploadMeterBar: document.querySelector("#uploadMeterBar"),
  startButton: document.querySelector("#startButton"),
  refreshJobs: document.querySelector("#refreshJobs"),
  jobsList: document.querySelector("#jobsList"),
  jobCount: document.querySelector("#jobCount"),
  serverStatus: document.querySelector("#serverStatus"),
  resultTitle: document.querySelector("#resultTitle"),
  transcriptBox: document.querySelector("#transcriptBox"),
  audioPlayer: document.querySelector("#audioPlayer"),
  copyButton: document.querySelector("#copyButton"),
  downloadButton: document.querySelector("#downloadButton"),
};

function createElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDate(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function setStatus(text, kind = "ready") {
  els.serverStatus.textContent = text;
  els.serverStatus.dataset.kind = kind;
}

function setUploadProgress(value) {
  els.uploadMeter.hidden = value <= 0 || value >= 1;
  els.uploadMeterBar.style.width = `${Math.round(value * 100)}%`;
}

async function loadConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) throw new Error("Could not load dashboard config.");
  const config = await response.json();
  els.model.textContent = "";
  const preferred = config.default_model || "large-v3";
  for (const model of config.models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    option.selected = model === preferred;
    els.model.appendChild(option);
  }
  els.chunkSeconds.value = config.default_chunk_seconds || 300;
}

async function loadJobs() {
  const response = await fetch("/api/jobs");
  if (!response.ok) throw new Error("Could not load jobs.");
  state.jobs = await response.json();
  renderJobs();
}

function renderSelectedFiles() {
  els.fileList.textContent = "";
  const files = [...els.files.files];
  if (!files.length) return;
  for (const file of files) {
    const row = createElement("div", "file-row");
    row.appendChild(createElement("div", "file-name", file.name));
    row.appendChild(createElement("div", "file-size", formatBytes(file.size)));
    els.fileList.appendChild(row);
  }
}

function renderJobs() {
  els.jobsList.textContent = "";
  els.jobCount.textContent = `${state.jobs.length} ${state.jobs.length === 1 ? "item" : "items"}`;

  if (!state.jobs.length) {
    els.jobsList.appendChild(createElement("div", "empty-state", "No jobs yet."));
    return;
  }

  for (const job of state.jobs) {
    const item = createElement("button", "job-item");
    item.type = "button";
    item.dataset.jobId = job.id;
    if (job.id === state.selectedJobId) item.classList.add("is-selected");

    const top = createElement("div", "job-top");
    top.appendChild(createElement("div", "job-title", job.original_filename));
    const label = job.status.charAt(0).toUpperCase() + job.status.slice(1);
    const status = createElement("span", `status ${job.status}`, label);
    top.appendChild(status);
    item.appendChild(top);

    const meta = createElement("div", "job-meta");
    const speakerMode = job.speaker_mode && job.speaker_mode !== "none" ? ` / ${job.speaker_mode}` : "";
    const model = `${job.model} / ${job.language || "auto"}${speakerMode}`;
    const chunks = job.total_chunks ? `${job.current_chunk || 0}/${job.total_chunks}` : "queued";
    meta.appendChild(createElement("span", "", model));
    meta.appendChild(createElement("span", "", `${formatBytes(job.size_bytes)} / ${chunks}`));
    meta.appendChild(createElement("span", "", formatDate(job.created_at)));
    item.appendChild(meta);

    const track = createElement("div", "progress-track");
    const bar = document.createElement("span");
    bar.style.width = `${Math.round((job.progress || 0) * 100)}%`;
    if (job.status === "error") bar.style.background = "var(--red)";
    if (job.status === "done") bar.style.background = "var(--green)";
    track.appendChild(bar);
    item.appendChild(track);

    const message = job.error || job.message || "";
    item.appendChild(createElement("div", "job-message", message));
    item.addEventListener("click", () => selectJob(job.id));
    els.jobsList.appendChild(item);
  }
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  renderJobs();

  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) return;
  const job = await response.json();
  state.selectedTranscript = job.transcript || "";
  els.resultTitle.textContent = job.original_filename;
  els.transcriptBox.textContent =
    job.transcript || job.error || "Transcript is not ready yet.";

  els.copyButton.disabled = !job.transcript;
  els.downloadButton.disabled = !job.download_url;
  els.downloadButton.dataset.url = job.download_url || "";

  if (job.audio_url) {
    els.audioPlayer.hidden = false;
    const nextAudioUrl = new URL(job.audio_url, window.location.origin).href;
    if (els.audioPlayer.src !== nextAudioUrl) {
      els.audioPlayer.src = job.audio_url;
    }
  } else {
    els.audioPlayer.hidden = true;
    els.audioPlayer.removeAttribute("src");
  }
}

function buildFormData() {
  const formData = new FormData();
  for (const file of els.files.files) {
    formData.append("audio", file);
  }
  formData.append("model", els.model.value);
  formData.append("language", els.language.value);
  formData.append("task", els.task.value);
  formData.append("chunk_seconds", els.chunkSeconds.value);
  formData.append("speaker_mode", els.speakerMode.value);
  formData.append("initial_prompt", els.initialPrompt.value.trim());
  if (els.includeTimestamps.checked) {
    formData.append("include_timestamps", "true");
  }
  return formData;
}

function uploadFiles() {
  if (!els.files.files.length) {
    setStatus("Choose files", "warning");
    return;
  }

  els.startButton.disabled = true;
  setStatus("Uploading", "busy");
  setUploadProgress(0.01);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/jobs");
  xhr.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      setUploadProgress(event.loaded / event.total);
    }
  });
  xhr.addEventListener("load", async () => {
    els.startButton.disabled = false;
    setUploadProgress(1);
    let payload = {};
    try {
      payload = JSON.parse(xhr.responseText || "{}");
    } catch {
      payload = {};
    }

    if (xhr.status >= 200 && xhr.status < 300) {
      setStatus("Queued", "ready");
      els.files.value = "";
      renderSelectedFiles();
      await loadJobs();
      if (payload.jobs && payload.jobs[0]) {
        selectJob(payload.jobs[0].id);
      }
    } else {
      setStatus("Upload failed", "error");
      els.transcriptBox.textContent = payload.error || "Upload failed.";
    }
  });
  xhr.addEventListener("error", () => {
    els.startButton.disabled = false;
    setUploadProgress(1);
    setStatus("Network error", "error");
  });
  xhr.send(buildFormData());
}

function installEvents() {
  els.chooseButton.addEventListener("click", () => els.files.click());
  els.files.addEventListener("change", renderSelectedFiles);
  els.refreshJobs.addEventListener("click", loadJobs);
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    uploadFiles();
  });

  for (const eventName of ["dragenter", "dragover"]) {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("is-over");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    els.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("is-over");
    });
  }
  els.dropzone.addEventListener("drop", (event) => {
    els.files.files = event.dataTransfer.files;
    renderSelectedFiles();
  });

  els.copyButton.addEventListener("click", async () => {
    if (!state.selectedTranscript) return;
    await navigator.clipboard.writeText(state.selectedTranscript);
    setStatus("Copied", "ready");
  });

  els.downloadButton.addEventListener("click", () => {
    const url = els.downloadButton.dataset.url;
    if (url) window.location.href = url;
  });
}

async function start() {
  try {
    await loadConfig();
    await loadJobs();
    installEvents();
    state.pollTimer = window.setInterval(async () => {
      await loadJobs();
      if (state.selectedJobId) {
        await selectJob(state.selectedJobId);
      }
    }, 2500);
    setStatus("Ready", "ready");
  } catch (error) {
    setStatus("Offline", "error");
    els.transcriptBox.textContent = error.message;
  }
}

start();
