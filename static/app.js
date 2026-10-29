const LEVEL_STYLE = {
  safe: { color: "#2dd4a8", label: "안전" },
  caution: { color: "#f4c64e", label: "조심" },
  danger: { color: "#ff5c68", label: "위험" },
};

const state = {
  videos: [], selectedVideo: null, selectedObservationId: null,
  observations: [], segments: [], viewMode: "video", supersededCount: 0,
};
const map = L.map("map", { zoomControl: false, preferCanvas: true }).setView([37.4775, 126.8773], 17);
L.control.zoom({ position: "bottomright" }).addTo(map);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 20,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const pointLayer = L.layerGroup().addTo(map);
const frameImage = document.getElementById("frame-image");
const frameWrap = frameImage.closest(".frame-wrap");

function fitFrameImage() {
  if (!frameImage.naturalWidth || !frameImage.naturalHeight) return;
  frameWrap.style.aspectRatio = `${frameImage.naturalWidth} / ${frameImage.naturalHeight}`;
  frameImage.style.width = "100%";
  frameImage.style.height = "100%";
}

frameImage.addEventListener("load", fitFrameImage);
const sourceVideo = document.getElementById("source-player");
sourceVideo.addEventListener("loadedmetadata", () => {
  if (sourceVideo.videoWidth && sourceVideo.videoHeight) {
    sourceVideo.style.aspectRatio = `${sourceVideo.videoWidth} / ${sourceVideo.videoHeight}`;
  }
});

function showToast(message, error = false) {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast show${error ? " error" : ""}`;
  window.setTimeout(() => { toast.className = "toast"; }, 3200);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* non-JSON */ }
  if (!response.ok) throw new Error(payload.error || `요청 실패 (${response.status})`);
  return payload;
}

function levelFromScore(score, grade = "") {
  if (grade.includes("DISCONNECTED")) return "danger";
  if (score <= 25) return "safe";
  if (score <= 50) return "caution";
  return "danger";
}

function routeGrade(score) {
  if (score >= 85) return "A등급 · 우수";
  if (score >= 70) return "B등급 · 양호";
  if (score >= 55) return "C등급 · 보통";
  if (score >= 40) return "D등급 · 미흡";
  return "E등급 · 개선 시급";
}

function formatStatus(video) {
  if (video.status === "queued") return "대기 중";
  if (video.status === "processing") return `분석 ${video.progress}%`;
  if (video.status === "completed") return "분석 완료";
  return "분석 실패";
}

function renderVideoList() {
  const list = document.getElementById("video-list");
  const allButton = document.getElementById("all-videos-button");
  allButton.classList.toggle("selected", state.viewMode === "all");
  document.getElementById("all-videos-count").textContent = `${state.videos.length}개 영상 · 최신 위치 우선`;
  list.replaceChildren();
  if (!state.videos.length) {
    const empty = document.createElement("p");
    empty.className = "empty-message";
    empty.textContent = "아직 저장된 점검 영상이 없습니다.";
    list.append(empty);
    return;
  }
  state.videos.forEach((video) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `video-item${state.viewMode === "video" && state.selectedVideo?.id === video.id ? " selected" : ""}`;
    const level = video.overall_level || "safe";
    button.innerHTML = `<i class="level-dot ${level}"></i><span><strong></strong><small></small></span><em>${video.safety_score == null ? "—" : `${Math.round(video.safety_score)}점`}</em>`;
    button.querySelector("strong").textContent = video.original_name;
    button.querySelector("small").textContent = `${formatStatus(video)} · ${new Date(video.created_at).toLocaleString("ko-KR")}`;
    button.addEventListener("click", () => selectVideo(video.id, true));
    list.append(button);
  });
}

function updateSummary() {
  if (state.viewMode === "all") {
    const risks = state.observations.map((item) => Number(item.risk_score) || 0);
    const safety = risks.length ? 100 - risks.reduce((sum, value) => sum + value, 0) / risks.length : null;
    const segmentKeys = new Set(state.observations.map((item) => `${item.video_id}:${item.segment_index}`));
    const riskSegmentKeys = new Set(state.observations.filter((item) => levelFromScore(item.risk_score, item.grade) !== "safe").map((item) => `${item.video_id}:${item.segment_index}`));
    const totalDistance = state.videos.reduce((sum, video) => sum + Number(video.distance_meters || 0), 0);
    const totalDuration = state.videos.reduce((sum, video) => sum + Number(video.duration_seconds || 0), 0);
    document.getElementById("safety-score").textContent = safety == null ? "—" : `${Math.round(safety)}점`;
    document.getElementById("safety-grade").textContent = risks.length ? `최신 GPS ${risks.length}개 · 이전 ${state.supersededCount}개 대체` : "표시할 분석 지점 없음";
    document.getElementById("distance-total").textContent = `${Math.round(totalDistance)}m`;
    document.getElementById("duration-total").textContent = `${Math.round(totalDuration)}초 누적`;
    document.getElementById("segment-count").textContent = segmentKeys.size;
    document.getElementById("risk-segment-count").textContent = riskSegmentKeys.size;
    document.getElementById("progress-wrap").hidden = true;
    document.getElementById("report-button").disabled = true;
    document.getElementById("report-download").hidden = true;
    return;
  }
  const video = state.selectedVideo;
  const safety = video?.safety_score;
  document.getElementById("safety-score").textContent = safety == null ? "—" : `${Math.round(safety)}점`;
  document.getElementById("safety-grade").textContent = safety == null ? (video ? formatStatus(video) : "영상 선택 전") : routeGrade(safety);
  document.getElementById("distance-total").textContent = `${Math.round(video?.distance_meters || 0)}m`;
  document.getElementById("duration-total").textContent = `${Math.round(video?.duration_seconds || 0)}초 영상`;
  document.getElementById("segment-count").textContent = state.segments.length;
  document.getElementById("risk-segment-count").textContent = state.segments.filter((segment) => segment.level !== "safe").length;
  const progressWrap = document.getElementById("progress-wrap");
  if (video && ["queued", "processing"].includes(video.status)) {
    progressWrap.hidden = false;
    setProgress(video.progress || 0, `AI 프레임 분석 · ${video.processed_frames || 0}/${video.total_frames || "?"}`);
  } else if (!document.getElementById("native-video-button").disabled) {
    progressWrap.hidden = true;
  }
  document.getElementById("report-button").disabled = video?.status !== "completed";
  const download = document.getElementById("report-download");
  download.hidden = !video?.report_url;
  if (video?.report_url) download.href = video.report_url;
}

function representativePoints(items) {
  const result = [];
  let lastSecond = -1;
  let lastLat = null;
  let lastLng = null;
  items.forEach((item) => {
    const second = Math.floor(item.video_seconds);
    const moved = lastLat === null || Math.abs(item.latitude - lastLat) + Math.abs(item.longitude - lastLng) > 0.000002;
    if (second !== lastSecond || moved) {
      result.push(item);
      lastSecond = second;
      lastLat = item.latitude;
      lastLng = item.longitude;
    }
  });
  return result;
}

function drawMap(fit = false) {
  routeLayer.clearLayers();
  pointLayer.clearLayers();
  const bySegment = new Map();
  state.observations.forEach((item) => {
    const key = `${item.video_id}:${item.segment_index}`;
    if (!bySegment.has(key)) bySegment.set(key, []);
    bySegment.get(key).push(item);
  });
  const bounds = [];
  bySegment.forEach((allItems) => {
    const items = representativePoints(allItems);
    if (!items.length) return;
    const segmentIndex = Number(items[0].segment_index);
    const segment = state.segments.find((entry) => entry.segment_index === segmentIndex);
    const measuredLevel = levelFromScore(Math.max(...items.map((item) => item.risk_score)));
    const level = segment?.frame_count ? segment.level : (items[0].segment_level || measuredLevel);
    const latLngs = items.map((item) => [item.latitude, item.longitude]);
    bounds.push(...latLngs);
    if (latLngs.length > 1) {
      L.polyline(latLngs, { color: "#07111f", weight: 9, opacity: .86, interactive: false }).addTo(routeLayer);
      const line = L.polyline(latLngs, { color: LEVEL_STYLE[level].color, weight: 5, opacity: .96 }).addTo(routeLayer);
      const videoLabel = state.viewMode === "all" ? `${items[0].video_name || `영상 #${items[0].video_id}`} · ` : "";
      line.bindTooltip(`${videoLabel}${segmentIndex + 1}구간 · ${LEVEL_STYLE[level].label}`, { sticky: true });
      line.on("click", (event) => {
        const nearest = items.reduce((best, item) => map.distance(event.latlng, [item.latitude, item.longitude]) < map.distance(event.latlng, [best.latitude, best.longitude]) ? item : best);
        selectObservation(nearest);
      });
    }
    items.forEach((item) => {
      const itemLevel = levelFromScore(item.risk_score, item.grade);
      const marker = L.circleMarker([item.latitude, item.longitude], { radius: itemLevel === "danger" ? 5 : 3.5, color: "#f8fafc", weight: 1, fillColor: LEVEL_STYLE[itemLevel].color, fillOpacity: .95 }).addTo(pointLayer);
      marker.on("click", () => selectObservation(item));
    });
  });
  if (fit && bounds.length) map.fitBounds(bounds, { padding: [35, 35], maxZoom: 18 });
}

function selectObservation(item) {
  state.selectedObservationId = item.id;
  const level = levelFromScore(item.risk_score, item.grade);
  const source = state.videos.find((video) => video.id === Number(item.video_id));
  document.getElementById("detail-title").textContent = `${item.video_name || source?.original_name || "영상"} · 프레임 #${item.frame_index}`;
  const badge = document.getElementById("risk-badge");
  badge.className = `risk-badge ${level}`;
  badge.textContent = `${LEVEL_STYLE[level].label} ${item.risk_score}`;
  frameImage.src = `${item.frame_url}?v=${item.id}`;
  frameImage.hidden = false;
  document.getElementById("frame-placeholder").hidden = true;
  document.getElementById("detail-segment").textContent = `${item.segment_index + 1}구간`;
  document.getElementById("detail-seconds").textContent = `${item.video_seconds.toFixed(1)}초 · #${item.frame_index}`;
  document.getElementById("detail-risk").textContent = `${item.risk_score}점 · ${item.grade}`;
  document.getElementById("detail-width").textContent = `${Math.round(item.effective_ratio * 100)}% (약 ${(2.4 * item.effective_ratio).toFixed(1)}m)`;
  document.getElementById("detail-gps").textContent = `${item.latitude.toFixed(6)}, ${item.longitude.toFixed(6)}`;
  document.getElementById("detail-time").textContent = item.captured_at || "SRT 시각 없음";
  const player = document.getElementById("source-player");
  if (source && player.dataset.videoId !== String(source.id)) {
    player.src = source.source_url;
    player.dataset.videoId = String(source.id);
    player.load();
  }
  const seek = () => { player.currentTime = Math.min(item.video_seconds, player.duration || item.video_seconds); };
  if (player.readyState >= 1) seek();
  else player.addEventListener("loadedmetadata", seek, { once: true });
}

async function selectVideo(videoId, fit = false) {
  const video = state.videos.find((item) => item.id === videoId);
  if (!video) return;
  state.viewMode = "video";
  if (state.selectedVideo?.id !== video.id) state.selectedObservationId = null;
  state.selectedVideo = video;
  document.getElementById("map-title").textContent = video.original_name;
  const sourcePlayer = document.getElementById("source-player");
  if (sourcePlayer.dataset.videoId !== String(video.id)) {
    sourcePlayer.src = video.source_url;
    sourcePlayer.dataset.videoId = String(video.id);
    sourcePlayer.load();
  }
  renderVideoList();
  try {
    const [observationData, segmentData] = await Promise.all([api(video.observations_url), api(video.segments_url)]);
    state.observations = observationData.observations;
    state.segments = segmentData.segments;
    updateSummary();
    drawMap(fit);
    if (state.observations.length) {
      selectObservation(
        state.observations.find((item) => item.id === state.selectedObservationId)
        || state.observations[0]
      );
    }
  } catch (error) { showToast(error.message, true); }
}

async function selectAllVideos(fit = false) {
  const previousObservationId = state.viewMode === "all" ? state.selectedObservationId : null;
  state.viewMode = "all";
  state.selectedVideo = null;
  state.selectedObservationId = previousObservationId;
  state.segments = [];
  document.getElementById("map-title").textContent = "전체 최신 GPS 포인트";
  renderVideoList();
  try {
    const data = await api("/api/map/observations");
    state.observations = data.observations;
    state.supersededCount = data.superseded_count || 0;
    updateSummary();
    drawMap(fit);
    if (state.observations.length) {
      selectObservation(
        state.observations.find((item) => item.id === state.selectedObservationId)
        || state.observations[0]
      );
    }
  } catch (error) { showToast(error.message, true); }
}

async function loadVideos({ keepSelection = true } = {}) {
  const previousId = keepSelection ? state.selectedVideo?.id : null;
  const keepAll = keepSelection && state.viewMode === "all";
  const { videos } = await api("/api/videos");
  state.videos = videos;
  if (keepAll && videos.length) {
    await selectAllVideos(false);
    return;
  }
  const nextId = videos.some((video) => video.id === previousId) ? previousId : videos[0]?.id;
  if (nextId) await selectVideo(nextId, !previousId);
  else {
    state.selectedVideo = null; state.observations = []; state.segments = [];
    renderVideoList(); updateSummary(); drawMap();
  }
}

function setProgress(value, label) {
  const progress = Math.max(0, Math.min(100, Number(value) || 0));
  document.getElementById("progress-wrap").hidden = false;
  document.getElementById("progress-label").textContent = label;
  document.getElementById("progress-value").textContent = `${Math.round(progress)}%`;
  document.getElementById("progress-bar").style.width = `${progress}%`;
}

document.getElementById("fit-button").addEventListener("click", () => drawMap(true));
document.getElementById("refresh-button").addEventListener("click", () => loadVideos().catch((error) => showToast(error.message, true)));
document.getElementById("all-videos-button").addEventListener("click", () => selectAllVideos(true));

function showMedia(kind) {
  const frameSelected = kind === "frame";
  document.getElementById("frame-view").hidden = !frameSelected;
  document.getElementById("video-view").hidden = frameSelected;
  document.getElementById("frame-tab").classList.toggle("active", frameSelected);
  document.getElementById("video-tab").classList.toggle("active", !frameSelected);
}
document.getElementById("frame-tab").addEventListener("click", () => showMedia("frame"));
document.getElementById("video-tab").addEventListener("click", () => showMedia("video"));

document.getElementById("native-video-button").addEventListener("click", async () => {
  const button = document.getElementById("native-video-button");
  button.disabled = true;
  button.className = "file-field video-picker busy";
  document.getElementById("selected-video-name").textContent = "로컬 파일 선택창을 확인하세요";
  document.getElementById("matched-srt-name").textContent = "영상을 고르면 SRT를 자동으로 찾습니다";
  try {
    const payload = await api("/api/videos/select-local", { method: "POST" });
    if (payload.cancelled) {
      document.getElementById("selected-video-name").textContent = "영상 파일을 선택하세요";
      document.getElementById("matched-srt-name").textContent = "같은 위치·같은 이름의 SRT가 자동 연결됩니다";
      button.className = "file-field video-picker";
      return;
    }
    button.className = "file-field video-picker matched";
    document.getElementById("selected-video-name").textContent = payload.video.original_name;
    document.getElementById("matched-srt-name").textContent = `${payload.srt_name} 자동 연결됨`;
    showToast(payload.duplicate ? "이미 분석 이력이 있는 영상입니다." : "영상 선택 완료 · AI 분석을 시작했습니다.");
    await loadVideos({ keepSelection: false });
  } catch (error) {
    button.className = "file-field video-picker failed";
    document.getElementById("selected-video-name").textContent = "영상을 선택하지 못했습니다";
    document.getElementById("matched-srt-name").textContent = error.message;
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
});

document.getElementById("report-button").addEventListener("click", async () => {
  if (!state.selectedVideo) return;
  const button = document.getElementById("report-button"); button.disabled = true; button.textContent = "리포트 작성 중…";
  try {
    const result = await api(`/api/videos/${state.selectedVideo.id}/report`, { method: "POST" });
    const download = document.getElementById("report-download"); download.href = result.download_url; download.hidden = false;
    showToast(result.message);
  } catch (error) { showToast(error.message, true); }
  finally { button.disabled = false; button.textContent = "점검 리포트 생성 · 저장"; }
});

async function initialize() {
  const status = document.getElementById("system-status");
  try {
    await api("/api/health"); status.className = "status connected"; status.lastElementChild.textContent = "MySQL 연결됨";
    await loadVideos({ keepSelection: false });
  } catch (error) {
    status.className = "status failed"; status.lastElementChild.textContent = "MySQL 연결 실패"; showToast(error.message, true);
  }
}

initialize();
window.setInterval(async () => {
  if (!state.videos.some((video) => ["queued", "processing"].includes(video.status))) return;
  try { await loadVideos(); } catch (_) { /* retry on next poll */ }
}, 3000);
