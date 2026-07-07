const test = require("node:test");
const assert = require("node:assert/strict");
const { buildPayload, parseErrorDetail, NUMERIC_FIELDS } = require("./script.js");

test("buildPayload converts numeric fields and leaves strings alone", () => {
  const entries = [
    ["Age", "22"],
    ["Gender", "Male"],
    ["CGPA", "8.1"],
  ];
  const payload = buildPayload(entries, NUMERIC_FIELDS);
  assert.equal(payload.Age, 22);
  assert.equal(typeof payload.Age, "number");
  assert.equal(payload.Gender, "Male");
  assert.equal(payload.CGPA, 8.1);
});

test("buildPayload throws a readable error on invalid numeric input", () => {
  const entries = [["Age", "not-a-number"]];
  assert.throws(() => buildPayload(entries, NUMERIC_FIELDS), /Age/);
});

test("buildPayload throws on blank numeric input", () => {
  const entries = [["Backlogs", ""]];
  assert.throws(() => buildPayload(entries, NUMERIC_FIELDS));
});

test("parseErrorDetail extracts FastAPI's single-string detail", () => {
  const raw = JSON.stringify({ detail: "CGPA must be <= 10" });
  assert.equal(parseErrorDetail(raw), "CGPA must be <= 10");
});

test("parseErrorDetail joins FastAPI's validation-error array", () => {
  const raw = JSON.stringify({
    detail: [
      { msg: "Input should be less than or equal to 10" },
      { msg: "Input should be greater than or equal to 0" },
    ],
  });
  assert.equal(
    parseErrorDetail(raw),
    "Input should be less than or equal to 10; Input should be greater than or equal to 0"
  );
});

test("parseErrorDetail falls back to raw text for non-JSON bodies", () => {
  assert.equal(parseErrorDetail("Internal Server Error"), "Internal Server Error");
});
