const NUMERIC_FIELDS = new Set([
  "Age", "CGPA", "Internships", "Projects", "Coding_Skills",
  "Communication_Skills", "Aptitude_Test_Score", "Soft_Skills_Rating",
  "Certifications", "Backlogs",
]);

// Pure -- no DOM access -- so it's unit-testable from Node (see script.test.js).
function buildPayload(entries, numericFields) {
  const payload = {};
  for (const [key, value] of entries) {
    if (!numericFields.has(key)) {
      payload[key] = value;
      continue;
    }
    const num = parseFloat(value);
    if (Number.isNaN(num)) {
      throw new Error(`Please enter a number for "${key.replaceAll("_", " ")}".`);
    }
    payload[key] = num;
  }
  return payload;
}

// Pure -- no DOM access -- so it's unit-testable from Node (see script.test.js).
function parseErrorDetail(raw) {
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.detail)) {
      return parsed.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    }
    if (typeof parsed.detail === "string") return parsed.detail;
    return raw;
  } catch {
    return raw;
  }
}

async function readableError(res) {
  return parseErrorDetail(await res.text());
}

function render(data) {
  const resultBox = document.getElementById("result");
  resultBox.className = data.prediction === "Placed" ? "placed" : "not-placed";
  resultBox.replaceChildren();

  if (!data || typeof data.probability_placed !== "number" || !Array.isArray(data.top_features)) {
    resultBox.textContent = "Received an unexpected response from the server.";
    return;
  }

  const pct = (data.probability_placed * 100).toFixed(1);

  const heading = document.createElement("h2");
  heading.textContent = `${data.prediction} (${pct}% placement probability)`;

  const label = document.createElement("p");
  label.textContent = "Top contributing factors:";

  const list = document.createElement("ul");
  for (const f of data.top_features) {
    const li = document.createElement("li");
    const sign = f.approx_probability_impact_pct > 0 ? "+" : "";
    li.textContent = `${f.explanation} (${sign}${f.approx_probability_impact_pct}% probability impact)`;
    list.appendChild(li);
  }

  resultBox.append(heading, label, list);
}

if (typeof document !== "undefined") {
  document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const resultBox = document.getElementById("result");

    let payload;
    try {
      payload = buildPayload(new FormData(form).entries(), NUMERIC_FIELDS);
    } catch (err) {
      resultBox.style.display = "block";
      resultBox.className = "not-placed";
      resultBox.textContent = `Error: ${err.message}`;
      return;
    }

    resultBox.style.display = "block";
    resultBox.className = "";
    resultBox.textContent = "Predicting...";

    try {
      const res = await fetch(`/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        throw new Error(await readableError(res));
      }
      const data = await res.json();
      render(data);
    } catch (err) {
      resultBox.className = "not-placed";
      resultBox.textContent = `Error: ${err.message}`;
    }
  });
}

// Node-only export for tests (browsers never define `module`).
if (typeof module !== "undefined" && module.exports) {
  module.exports = { buildPayload, parseErrorDetail, NUMERIC_FIELDS };
}
