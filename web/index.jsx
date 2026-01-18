const Icon = ({ path, color = "currentColor", size = 24, className = "" }) => (
    <svg 
        xmlns="http://www.w3.org/2000/svg" 
        width={size} 
        height={size} 
        viewBox="0 0 24 24" 
        fill="none" 
        stroke={color} 
        strokeWidth="2" 
        strokeLinecap="round" 
        strokeLinejoin="round" 
        className={className}
    >
        {path}
    </svg>
);

const Icons = {
    Activity: (props) => <Icon {...props} path={<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>} />,
    Server: (props) => <Icon {...props} path={<><rect x="2" y="2" width="20" height="8" rx="2" ry="2"></rect><rect x="2" y="14" width="20" height="8" rx="2" ry="2"></rect><line x1="6" y1="6" x2="6.01" y2="6"></line><line x1="6" y1="18" x2="6.01" y2="18"></line></>} />,
    CheckCircle: (props) => <Icon {...props} path={<><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></>} />,
    AlertCircle: (props) => <Icon {...props} path={<><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></>} />,
    Upload: (props) => <Icon {...props} path={<><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></>} />,
    FileText: (props) => <Icon {...props} path={<><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></>} />,
    TrendingUp: (props) => <Icon {...props} path={<><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></>} />,
    Play: (props) => <Icon {...props} path={<polygon points="5 3 19 12 5 21 5 3"></polygon>} />,
    Terminal: (props) => <Icon {...props} path={<><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></>} />,
    FileBarChart: (props) => <Icon {...props} path={<><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><path d="M12 18v-6"></path><path d="M8 18v-1"></path><path d="M16 18v-3"></path></>} />,
    Download: (props) => <Icon {...props} path={<><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></>} />
};

const { useState, useEffect, useRef } = React;

function App() {
    const [logs, setLogs] = useState([]);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [reportMarkdown, setReportMarkdown] = useState('');
    const [analysisResults, setAnalysisResults] = useState(null);
    const [activeTab, setActiveTab] = useState('console');
    const [backendStatus, setBackendStatus] = useState('unknown');
    const [uploadedFiles, setUploadedFiles] = useState([]);

    const consoleEndRef = useRef(null);

    // Auto-scroll console
    useEffect(() => {
        consoleEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [logs]);

    // Check Backend Health on Mount
    useEffect(() => {
        checkBackendHealth();
        const interval = setInterval(checkBackendHealth, 10000);
        return () => clearInterval(interval);
    }, []);

    const checkBackendHealth = async () => {
        try {
            const res = await fetch('http://localhost:5000/health');
            if (res.ok) {
                setBackendStatus('connected');
            } else {
                setBackendStatus('error');
            }
        } catch (e) {
            setBackendStatus('error');
        }
    };

    const addLog = (msg) => {
        const time = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, `[${time}] ${msg}`]);
    };

    const runAnalysis = async () => {
        if (backendStatus !== 'connected') {
            addLog("ERROR: Backend not connected. Please run 'python backend.py' in your terminal.");
            return;
        }

        if (uploadedFiles.length === 0) {
            addLog("ERROR: No files selected. Please upload CSV data files.");
            return;
        }

        setIsProcessing(true);
        setProgress(0);
        setLogs([]);
        setAnalysisResults(null);
        setReportMarkdown('');
        setActiveTab('console');

        addLog("=== Starting Analysis (Remote Python Kernel) ===");
        
        const formData = new FormData();
        Array.from(uploadedFiles).forEach(file => {
            formData.append('files', file);
        });

        try {
            setProgress(20);
            addLog(`[Upload] Sending ${uploadedFiles.length} files to Python Backend...`);
            
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server returned ${response.status}`);
            }

            const data = await response.json();
            setProgress(80);
            
            if (data.logs) {
                data.logs.forEach(l => addLog(`[PYTHON] ${l}`));
            }

            if (data.results) {
                setAnalysisResults(data.results);
                const report = generateMarkdown(data.results, data.meta);
                setReportMarkdown(report);
                addLog("Analysis Successful. Data received.");
                setProgress(100);
                setTimeout(() => setActiveTab('dashboard'), 500);
            }

        } catch (error) {
            addLog(`CRITICAL ERROR: ${error.message}`);
            setProgress(0);
        } finally {
            setIsProcessing(false);
        }
    };

    const generateMarkdown = (data, meta) => {
        const timestamp = new Date().toLocaleString();
        let md = `# Aadhar Migration Prediction Report\n`;
        md += `**Generated:** ${timestamp}\n\n`;
        md += `**Backend Source:** Python/Pandas Analysis\n\n`;
        if (meta) {
            md += `**Data processed:** ${meta.total_records} records\n\n`;
            md += `**News Articles Analyzed:** ${meta.news_count}\n\n`;
        }
        md += `\n---\n\n`;
        md += `## Executive Summary\n\n`;
        
        if (data.length > 0) {
            const top = data[0];
            md += `Based on the backend trend analysis, **${top.state}** shows the highest projected migration activity `;
            md += `with a probability of **${top.probability}%**.\n\n`;
            md += `- **Primary Driver:** ${top.biasReason}\n`;
            md += `- **News Sentiment:** ${top.newsSentiment} (${top.newsAdj}x factor)\n`;
        }
        
        md += `\n## Integrated Prediction Ranking\n\n`;
        md += `| Rank | State | Base Pred | News Adj | Final Pred | Prob | Reason |\n`;
        md += `|---|---|---|---|---|---|---|\n`;
        
        data.forEach((row, index) => {
            md += `| ${index + 1} | ${row.state} | ${row.basePred} | ${row.newsAdj}x | ${row.finalPred} | ${row.probability}% | ${row.biasReason} |\n`;
        });
        
        return md;
    };

    const handleFileUpload = (e) => {
        const files = e.target.files;
        if (files.length === 0) return;
        setUploadedFiles(files);
        addLog(`Selected ${files.length} files ready for upload.`);
    };

    const downloadReport = () => {
        const blob = new Blob([reportMarkdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'Migration_Report.md';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    };

    // Parse Markdown safely
    const renderReportContent = () => {
        if (!reportMarkdown) return null;
        const htmlContent = marked.parse(reportMarkdown);
        return { __html: htmlContent };
    };

    return (
        <div className="min-h-screen flex flex-col">
            <header className="bg-blue-900 text-white p-4 shadow-lg flex items-center justify-between">
                <div className="flex items-center space-x-3">
                    <Icons.Activity className="h-6 w-6 text-blue-300" />
                    <h1 className="text-xl font-bold tracking-wide">Aadhar Migration Predictor</h1>
                </div>
                <div className="flex items-center space-x-2 bg-blue-800 rounded-full px-3 py-1 text-xs transition-colors">
                    <Icons.Server className={`h-3 w-3 ${backendStatus === 'connected' ? 'text-green-400' : 'text-red-400'}`} />
                    <span>Server: {backendStatus === 'connected' ? 'Online' : 'Offline'}</span>
                </div>
            </header>

            <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-1 space-y-6">
                    <div className={`rounded-xl shadow-sm border p-4 ${backendStatus === 'connected' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                        <div className="flex items-start">
                            {backendStatus === 'connected' ? (
                                <Icons.CheckCircle className="h-5 w-5 text-green-600 mr-2 mt-0.5" />
                            ) : (
                                <Icons.AlertCircle className="h-5 w-5 text-red-600 mr-2 mt-0.5" />
                            )}
                            <div>
                                <h2 className={`text-sm font-bold ${backendStatus === 'connected' ? 'text-green-800' : 'text-red-800'}`}>
                                    {backendStatus === 'connected' ? 'Backend Connected' : 'Backend Disconnected'}
                                </h2>
                                <p className={`text-xs mt-1 ${backendStatus === 'connected' ? 'text-green-700' : 'text-red-700'}`}>
                                    {backendStatus === 'connected' 
                                        ? 'Python flask server is ready.' 
                                        : 'Ensure "backend.py" is running on port 5000.'}
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
                        <h2 className="text-lg font-semibold mb-4 flex items-center">
                            <Icons.Upload className="h-5 w-5 mr-2 text-blue-600" /> Upload Data
                        </h2>
                        
                        <div className="space-y-4">
                            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:bg-gray-50 transition-colors relative">
                                <input 
                                    type="file" 
                                    multiple 
                                    accept=".csv"
                                    onChange={handleFileUpload}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                />
                                <Icons.FileText className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                                <p className="text-sm text-gray-600 font-medium">Drop CSV files here</p>
                                <p className="text-xs text-gray-400">or click to browse</p>
                            </div>

                            <div className="flex items-center justify-between text-sm text-gray-600">
                                <span>Files Selected:</span>
                                <span className="font-mono bg-gray-100 px-2 py-1 rounded">{uploadedFiles.length}</span>
                            </div>
                        </div>
                    </div>

                    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
                        <h2 className="text-lg font-semibold mb-4 flex items-center">
                            <Icons.TrendingUp className="h-5 w-5 mr-2 text-green-600" /> Run Analysis
                        </h2>
                        
                        <button 
                            onClick={runAnalysis}
                            disabled={isProcessing || backendStatus !== 'connected'}
                            className={`w-full py-3 px-4 rounded-lg text-white font-medium flex items-center justify-center space-x-2 transition-all ${
                                isProcessing || backendStatus !== 'connected'
                                ? 'bg-gray-400 cursor-not-allowed' 
                                : 'bg-blue-600 hover:bg-blue-700 hover:shadow-md'
                            }`}
                        >
                            {isProcessing ? (
                                <span className="flex items-center">
                                    <span className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></span>
                                    Processing... {progress}%
                                </span>
                            ) : (
                                <span className="flex items-center">
                                    <Icons.Play className="h-4 w-4 mr-2" />
                                    Execute Analysis
                                </span>
                            )}
                        </button>
                    </div>
                </div>

                <div className="lg:col-span-2 flex flex-col space-y-4">
                    <div className="flex space-x-1 bg-gray-200 p-1 rounded-lg w-fit">
                        <button 
                            onClick={() => setActiveTab('console')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'console' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-600 hover:text-gray-900'}`}
                        >
                            Console
                        </button>
                        <button 
                            onClick={() => setActiveTab('dashboard')}
                            disabled={!analysisResults}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'dashboard' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-600 hover:text-gray-900 disabled:opacity-50'}`}
                        >
                            Results
                        </button>
                        <button 
                            onClick={() => setActiveTab('report')}
                            disabled={!reportMarkdown}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${activeTab === 'report' ? 'bg-white text-blue-700 shadow-sm' : 'text-gray-600 hover:text-gray-900 disabled:opacity-50'}`}
                        >
                            Report
                        </button>
                    </div>

                    {activeTab === 'console' && (
                        <div className="flex-1 bg-gray-900 rounded-xl shadow-inner p-4 font-mono text-sm overflow-hidden flex flex-col h-[500px]">
                            <div className="flex items-center justify-between border-b border-gray-700 pb-2 mb-2">
                                <span className="text-gray-400 flex items-center"><Icons.Terminal className="h-4 w-4 mr-2"/> CLI Output</span>
                                <div className="flex space-x-2">
                                    <div className={`h-3 w-3 rounded-full ${backendStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'}`}></div>
                                </div>
                            </div>
                            <div className="flex-1 overflow-y-auto space-y-1 custom-scrollbar">
                                {logs.length === 0 && <span className="text-gray-600 italic">Waiting for input...</span>}
                                {logs.map((log, i) => (
                                    <div key={i} className="text-green-400 break-words font-light">
                                        <span className="text-green-700 mr-2">{'>'}</span>{log}
                                    </div>
                                ))}
                                <div ref={consoleEndRef} />
                            </div>
                        </div>
                    )}

                    {activeTab === 'dashboard' && analysisResults && (
                        <div className="flex-1 bg-white rounded-xl shadow-sm border border-gray-200 p-0 overflow-hidden h-[500px] flex flex-col">
                            <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                                <h3 className="font-semibold text-gray-700 flex items-center">
                                    <Icons.FileBarChart className="h-4 w-4 mr-2" /> Prediction Matrix
                                </h3>
                            </div>
                            <div className="overflow-auto flex-1 p-0">
                                <table className="w-full text-left text-sm">
                                    <thead className="bg-gray-50 sticky top-0">
                                        <tr>
                                            <th className="px-6 py-3 font-medium text-gray-500">Rank</th>
                                            <th className="px-6 py-3 font-medium text-gray-500">State</th>
                                            <th className="px-6 py-3 font-medium text-gray-500 text-right">Base Pred</th>
                                            <th className="px-6 py-3 font-medium text-gray-500 text-center">News Factor</th>
                                            <th className="px-6 py-3 font-medium text-gray-500 text-right">Final Score</th>
                                            <th className="px-6 py-3 font-medium text-gray-500 text-right">Probability</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-100">
                                        {analysisResults.map((row, i) => (
                                            <tr key={i} className="hover:bg-blue-50 transition-colors">
                                                <td className="px-6 py-3 font-bold text-gray-400">#{i+1}</td>
                                                <td className="px-6 py-3 font-medium text-gray-800">{row.state}</td>
                                                <td className="px-6 py-3 text-right text-gray-600">{row.basePred.toLocaleString()}</td>
                                                <td className="px-6 py-3 text-center">
                                                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                                                        row.newsAdj > 1 ? 'bg-green-100 text-green-700' : (row.newsAdj < 1 ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600')
                                                    }`}>
                                                        {row.newsAdj}x
                                                    </span>
                                                </td>
                                                <td className="px-6 py-3 text-right font-mono font-bold text-blue-600">{row.finalPred.toLocaleString()}</td>
                                                <td className="px-6 py-3 text-right">{row.probability}%</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {activeTab === 'report' && reportMarkdown && (
                        <div className="flex-1 bg-white rounded-xl shadow-sm border border-gray-200 flex flex-col h-[500px]">
                            <div className="p-3 border-b border-gray-100 flex justify-between items-center bg-gray-50 rounded-t-xl">
                                <span className="text-sm font-semibold text-gray-600 flex items-center">
                                    <Icons.FileText className="h-4 w-4 mr-2" /> Generated Report Preview
                                </span>
                                <button 
                                    onClick={downloadReport}
                                    className="flex items-center px-3 py-1.5 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
                                >
                                    <Icons.Download className="h-3 w-3 mr-1.5" /> Download .md
                                </button>
                            </div>
                            <div className="flex-1 p-8 overflow-y-auto bg-white font-serif text-gray-800 leading-relaxed">
                                {/* Updated Markdown Renderer using Marked.js */}
                                <div 
                                    className="markdown-body max-w-2xl mx-auto"
                                    dangerouslySetInnerHTML={renderReportContent()} 
                                />
                            </div>
                        </div>
                    )}

                </div>
            </main>
        </div>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);