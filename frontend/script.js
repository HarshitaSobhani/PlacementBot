document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const numericFields = new Set([
    "Age", "CGPA", "Internships", "Projects", "Coding_Skills",
    "Communication_Skills", "Aptitude_Test_Score", "Soft_Skills_Rating",
    "Certifications", "Backlogs",
  ]);

  const payload = {};
  new FormData(form).forEach((value, key) => {
    payload[key] = numericFields.has(key) ? parseFloat(value) : value;
  });

  const resultBox = document.getElementById("result");
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
      const err = await res.text();
      throw new Error(err);
    }
    const data = await res.json();
    render(data);
  } catch (err) {
    resultBox.className = "not-placed";
    resultBox.textContent = `Error: ${err.message}`;
  }
});

function render(data) {
  const resultBox = document.getElementById("result");
  resultBox.className = data.prediction === "Placed" ? "placed" : "not-placed";

  const pct = (data.probability_placed * 100).toFixed(1);
  const factors = data.top_features
    .map((f) => `<li>${f.explanation} (${f.approx_probability_impact_pct > 0 ? "+" : ""}${f.approx_probability_impact_pct}% probability impact)</li>`)
    .join("");

  resultBox.innerHTML = `
    <h2>${data.prediction} (${pct}% placement probability)</h2>
    <p>Top contributing factors:</p>
    <ul>${factors}</ul>
  `;
}
