const { spawn } = require("child_process");
const path = require("path");
const PYTHON_SCRIPT = path.join(__dirname, "../../healthyApp/api/exercise_api_script.py");

function runPython(inputData) {
  return new Promise((resolve, reject) => {
    const py = spawn("python", [PYTHON_SCRIPT]);
    let output = "";
    let error = "";

    py.stdout.on("data", (data) => output += data.toString());
    py.stderr.on("data", (data) => error += data.toString());

    py.on("close", (code) => {
      if (code !== 0) return reject(error || `Python exited with code ${code}`);
      try {
        resolve(JSON.parse(output));
      } catch (e) {
        reject(e);
      }
    });

    py.stdin.write(JSON.stringify(inputData));
    py.stdin.end();
  });
}

module.exports = { runPython };
