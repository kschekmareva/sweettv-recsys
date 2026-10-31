// Headless run of web/checks.js in Node: node web/checks_node.js
// (reads web/data/web_data.json and outputs/submissions/submission_STEP7_all.csv from disk)
const fs = require("fs"), path = require("path");
const C = require("./checks.js");
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "web_data.json"), "utf8"));
const csv = async () => fs.readFileSync(path.join(__dirname, "..", "outputs", "submissions", "submission_STEP7_all.csv"), "utf8");
C.runChecks(D, csv, console).then(r => process.exit(r.fail ? 1 : 0));
