// Where the model API lives. Default: same origin (/predict).
// Override without editing: open the page with ?api=https://your-backend.example.com/predict
const API_URL = new URLSearchParams(location.search).get("api") || "/predict";
const MIN_WORDS = 5;

const form = document.getElementById("form");
const titleEl = document.getElementById("title");
const bodyEl = document.getElementById("body");
const goBtn = document.getElementById("go");
const countEl = document.getElementById("count");
const stampEl = document.getElementById("stamp");
const verdictEl = document.getElementById("verdict");

const EXAMPLES = [
  {
    title: "SHOCKING: Secret memo proves senators hid the truth from you",
    body: "You will not believe what they are hiding. Insiders say the mainstream media will never report this, and patriots everywhere are demanding answers. Share this before it gets deleted!",
  },
  {
    title: "Parliament passes budget bill after late-night vote",
    body: "NEW DELHI (Reuters) - Lawmakers approved the annual budget bill early on Thursday after a late-night session, officials said, sending the measure to the upper house for final consideration.",
  },
];
let exampleIndex = 0;

function wordCount(text) {
  const t = text.trim();
  return t ? t.split(/\s+/).length : 0;
}

function updateCount() {
  const n = wordCount(titleEl.value + " " + bodyEl.value);
  countEl.textContent = n + (n === 1 ? " word" : " words");
}

function showEmpty() {
  verdictEl.classList.add("is-empty");
  verdictEl.innerHTML =
    '<p class="step-label">02 <span>/</span> ANALYSIS</p>' +
    "<h2>Your result will appear here</h2>" +
    '<p class="empty-copy">Add an article to see the model’s label and estimated class probabilities.</p>' +
    '<div class="empty-orbit" aria-hidden="true"><span></span><span></span><span></span></div>' +
    '<p class="empty-hint"><span class="hint-dot" aria-hidden="true"></span> Ready when you are</p>';
}

function showMessage(msg, isError) {
  verdictEl.classList.remove("is-empty");
  verdictEl.innerHTML = "<h2>Result</h2>";
  const p = document.createElement("p");
  p.textContent = msg;
  if (isError) p.className = "error";
  verdictEl.appendChild(p);
}

function meter(kind, value) {
  const pct = Math.round(value * 1000) / 10;
  return (
    '<div class="meter ' + kind + '">' +
    '<div class="row"><span>' + kind.charAt(0).toUpperCase() + kind.slice(1) +
    "</span><span>" + pct.toFixed(1) + "%</span></div>" +
    '<div class="track"><div class="fill" data-w="' + pct + '"></div></div></div>'
  );
}

function showResult(result) {
  verdictEl.classList.remove("is-empty");
  const label = result.label === "real" ? "real" : "fake";
  const probs = result.probabilities || {};
  const fake = probs.fake ?? (label === "fake" ? result.confidence : 1 - result.confidence);
  const real = probs.real ?? 1 - fake;

  // Stamp slams onto the sheet
  stampEl.className = "stamp " + label;
  stampEl.textContent = label;
  void stampEl.offsetWidth; // restart animation
  stampEl.classList.add("on");

  verdictEl.innerHTML =
    "<h2>Result</h2>" +
    '<div class="label-line ' + label + '">Labeled <span>' + label + "</span></div>" +
    meter("fake", fake) +
    meter("real", real) +
    '<p class="note">These are the model\'s estimated class probabilities, not a measure of whether the article is true.</p>';

  requestAnimationFrame(() => {
    verdictEl.querySelectorAll(".fill").forEach((el) => {
      el.style.width = el.dataset.w + "%";
    });
  });
}

async function check(event) {
  event.preventDefault();
  const text = (titleEl.value + " " + bodyEl.value).trim();

  if (wordCount(text) < MIN_WORDS) {
    showMessage("Add a headline and some article text first. A few words is too little for the model to judge.", true);
    return;
  }

  goBtn.disabled = true;
  goBtn.innerHTML = "<span>Analyzing…</span><span aria-hidden=\"true\">…</span>";
  stampEl.classList.remove("on");
  showMessage("Waiting for the model. The first request can take a while if the server was asleep.", false);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 45000);

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
      signal: controller.signal,
    });
    if (!res.ok) throw new Error("Server returned " + res.status);
    showResult(await res.json());
  } catch (err) {
    const timedOut = err.name === "AbortError";
    showMessage(
      timedOut
        ? "The model took too long to respond. Try again in a moment."
        : "Could not reach the model at " + API_URL + ". Check that the backend is running, then try again.",
      true
    );
  } finally {
    clearTimeout(timer);
    goBtn.disabled = false;
    goBtn.innerHTML = '<span>Analyze article</span><span aria-hidden="true">→</span>';
  }
}

document.getElementById("example").addEventListener("click", () => {
  const ex = EXAMPLES[exampleIndex++ % EXAMPLES.length];
  titleEl.value = ex.title;
  bodyEl.value = ex.body;
  stampEl.classList.remove("on");
  updateCount();
  showEmpty();
});

titleEl.addEventListener("input", updateCount);
bodyEl.addEventListener("input", updateCount);
form.addEventListener("submit", check);

updateCount();
showEmpty();
