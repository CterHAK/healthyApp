const os = require('os');const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const PYTHON_SCRIPT_PATH = path.join(__dirname, '..', 'api', '../../healthyApp/api/food_api_script.py');

const searchFood = async (req, res) => {
    try {
        const { q } = req.query;
        if (!q || typeof q !== 'string' || q.trim().length === 0) {
            return res.status(400).json({ error: "Query parameter 'q' must be a non-empty string" });
        }

        console.log(`Processing query: ${q}, Timestamp: ${new Date().toISOString()}`);
        const tempFile = path.join(os.tmpdir(), `search_${Date.now()}.json`);
        fs.writeFileSync(tempFile, JSON.stringify({ query: q.trim() }, null, 2), { encoding: 'utf8' });

        // Set SSE headers
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.flushHeaders();

        const pythonProcess = spawn('python', [
            PYTHON_SCRIPT_PATH,
            '--search',
            '--input-file',
            tempFile
        ], { timeout: 180000, encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });

        let outputData = '';
        let errorData = '';

        pythonProcess.stdout.setEncoding('utf8');
        let buffer = '';

        pythonProcess.stdout.on('data', (data) => {
        buffer += data;
        let lines = buffer.split('\n');
        buffer = lines.pop(); // giữ lại phần chưa hoàn chỉnh

        for (const line of lines) {
            if (!line.trim()) continue;
            try {
            const msg = JSON.parse(line);

            if (msg.type === 'chunk') {
                // gửi từng đoạn sang frontend
                res.write(`data: ${JSON.stringify({ chunk: msg.content })}\n\n`);
            } else if (msg.type === 'complete') {
                res.write(`data: ${JSON.stringify({ complete: true, summary: msg.summary })}\n\n`);
                res.end();
            } else if (msg.type === 'error') {
                res.write(`data: ${JSON.stringify({ error: msg.error })}\n\n`);
                res.end();
            }
            } catch (err) {
            console.error('Invalid JSON from Python:', line);
            }
        }
        });

        pythonProcess.stderr.on('data', (data) => {
            errorData += data.toString();
            console.error('Python stderr:', data.toString());
            res.write(`data: ${JSON.stringify({ stderr: data.toString() })}\n\n`);
        });

        pythonProcess.on('close', (code) => {
            try { fs.unlinkSync(tempFile); } catch (e) {}
            console.log(`Python process exited with code: ${code}`);
            if (code !== 0) {
                res.write(`data: ${JSON.stringify({ error: "Search failed", details: errorData })}\n\n`);
                res.end();
                return;
            }
            try {
                const jsonMatch = outputData.match(/{[\s\S]*}/);
                if (!jsonMatch) {
                    throw new Error('No valid JSON found in output');
                }
                const result = JSON.parse(jsonMatch[0]);
                if (result.error) {
                    res.write(`data: ${JSON.stringify({ error: "Search failed", details: result.error })}\n\n`);
                    res.end();
                    return;
                }
                res.write(`data: ${JSON.stringify({ complete: true, summary: result.summary })}\n\n`);
                res.end();
            } catch (e) {
                console.error('Failed to parse search results:', e, 'Output:', outputData, 'Error:', errorData);
                res.write(`data: ${JSON.stringify({ error: "Invalid search results", details: e.message })}\n\n`);
                res.end();
            }
        });

        pythonProcess.on('error', (err) => {
            console.error('Python process error:', err);
            try { fs.unlinkSync(tempFile); } catch (e) {}
            res.write(`data: ${JSON.stringify({ error: "Python process failed", details: err.message })}\n\n`);
            res.end();
        });

        req.on('close', () => {
            pythonProcess.kill();
            res.end();
        });
    } catch (error) {
        console.error('Error:', error);
        res.write(`data: ${JSON.stringify({ error: error.message })}\n\n`);
        res.end();
    }
};

module.exports = { searchFood };