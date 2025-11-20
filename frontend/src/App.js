import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [instruction, setInstruction] = useState('');
  const [configOutput, setConfigOutput] = useState('');
  const [references, setReferences] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showReferences, setShowReferences] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [loadingFiles, setLoadingFiles] = useState(false);
  const [showRetrievedContent, setShowRetrievedContent] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [retrievedContext, setRetrievedContext] = useState([]);
  const [retrievedContextDetailed, setRetrievedContextDetailed] = useState([]);
  const [fullPrompt, setFullPrompt] = useState('');
  const [selectedChunk, setSelectedChunk] = useState(null);

  // Fetch uploaded files from database
  const fetchUploadedFiles = async () => {
    setLoadingFiles(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/files`);
      const data = await response.json();
      if (response.ok) {
        setUploadedFiles(data.files || []);
      }
    } catch (error) {
      console.error('Error fetching files:', error);
    } finally {
      setLoadingFiles(false);
    }
  };

  // Fetch files on component mount
  useEffect(() => {
    fetchUploadedFiles();
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setUploadStatus('');
  };

  const handleUpload = async () => {
    if (!file) {
      setUploadStatus('Please select a file');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setUploadStatus('Uploading...');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/ingest`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setUploadStatus(`✅ Ingested ${data.chunks} chunks from ${data.source_id}`);
        setFile(null);
        document.getElementById('file-input').value = '';
        // Refresh the uploaded files list
        fetchUploadedFiles();
      } else {
        setUploadStatus(`❌ Error: ${data.error}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!instruction.trim()) {
      alert('Please enter an instruction');
      return;
    }

    setLoading(true);
    setConfigOutput('');
    setReferences([]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-config`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ instruction }),
      });

      const data = await response.json();

      if (response.ok) {
        setConfigOutput(data.config_output);
        setReferences(data.references || []);
        setRetrievedContext(data.context_chunks || []);
        setRetrievedContextDetailed(data.context_chunks_detailed || []);
        setFullPrompt(data.prompt_used || '');
      } else {
        setConfigOutput(`Error: ${data.error || 'Unknown error'}`);
        if (data.suggestion) {
          setConfigOutput(`${data.error}\n\n${data.suggestion}`);
        }
      }
    } catch (error) {
      setConfigOutput(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>Cisco Agent Configuration Generator</h1>
          <p style={{ fontSize: '1rem', marginTop: '10px' }}></p>
          <p>Upload PDFs, Excel, Word docs, code files, and more to generate configurations using AI</p>
        </header>

        <div className="content">
          {/* Upload Section */}
          <section className="section">
            <h2>📤 Upload Specifications</h2>
            <div className="upload-area">
              <input
                id="file-input"
                type="file"
                accept=".pdf,.xlsx,.xls,.docx,.txt,.sql,.py,.js,.html,.css,.json,.yaml,.yml,.md"
                onChange={handleFileChange}
                className="file-input"
              />
              <button
                onClick={handleUpload}
                disabled={loading || !file}
                className="btn btn-primary"
              >
                {loading ? 'Uploading...' : 'Upload & Ingest'}
              </button>
            </div>
            {uploadStatus && (
              <div className={`status ${uploadStatus.includes('✅') ? 'success' : 'error'}`}>
                {uploadStatus}
              </div>
            )}
            
            {/* Uploaded Files Section */}
            <div className="files-section">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                <h3 style={{ color: '#049FD9', margin: 0 }}>📁 Uploaded Files</h3>
                <button onClick={fetchUploadedFiles} className="refresh-btn" disabled={loadingFiles}>
                  {loadingFiles ? 'Loading...' : '🔄 Refresh'}
                </button>
              </div>
              
              {loadingFiles ? (
                <div className="empty-state">
                  <p>Loading files...</p>
                </div>
              ) : uploadedFiles.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon">📄</div>
                  <p>No files uploaded yet. Upload a file to get started!</p>
                </div>
              ) : (
                <div className="files-list">
                  {uploadedFiles.map((file, idx) => (
                    <div key={idx} className="file-card">
                      <div className="file-name">{file.source_id}</div>
                      <div className="file-meta">
                        <span className="file-type-badge">{file.file_type?.toUpperCase() || 'FILE'}</span>
                        <span className="chunks-count">{file.chunks_count} chunks</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>

          {/* Config Generator Section */}
          <section className="section">
            <h2>⚙️ Generate Configuration</h2>
            <div className="generator-area">
              <textarea
                value={instruction}
                onChange={(e) => setInstruction(e.target.value)}
                placeholder="Enter your instruction, e.g., 'Generate YAML for command X' or 'Create firewall config for rule Y'"
                className="instruction-input"
                rows="4"
              />
              <button
                onClick={handleGenerate}
                disabled={loading || !instruction.trim()}
                className="btn btn-primary"
              >
                {loading ? 'Generating...' : 'Generate Config'}
              </button>
            </div>

            {configOutput && (
              <div className="output-area">
                <h3>Generated Configuration:</h3>
                <pre className="code-block">
                  <code>{configOutput}</code>
                </pre>
                
                {/* Retrieved Context Dropdown */}
                {retrievedContextDetailed.length > 0 && (
                  <div className="references-section" style={{ marginTop: '20px' }}>
                    <div style={{ 
                      border: '1px solid #049FD9', 
                      borderRadius: '8px', 
                      overflow: 'hidden',
                      background: 'white'
                    }}>
                      <div 
                        onClick={() => setShowRetrievedContent(!showRetrievedContent)}
                        style={{
                          padding: '15px',
                          background: '#049FD9',
                          color: 'white',
                          cursor: 'pointer',
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          fontWeight: '600'
                        }}
                      >
                        <span>📚 Retrieved Context from Database ({retrievedContextDetailed.length} chunks)</span>
                        <span style={{ fontSize: '1.2rem' }}>
                          {showRetrievedContent ? '▼' : '▶'}
                        </span>
                      </div>
                      
                      {showRetrievedContent && (
                        <div style={{ padding: '15px', maxHeight: '500px', overflowY: 'auto' }}>
                          {retrievedContextDetailed.map((chunk, idx) => (
                            <div 
                              key={idx} 
                              style={{
                                marginBottom: '15px',
                                border: '1px solid #e0e0e0',
                                borderRadius: '6px',
                                overflow: 'hidden'
                              }}
                            >
                              <div
                                onClick={() => setSelectedChunk(selectedChunk === idx ? null : idx)}
                                style={{
                                  padding: '12px',
                                  background: '#f8f9fa',
                                  cursor: 'pointer',
                                  display: 'flex',
                                  justifyContent: 'space-between',
                                  alignItems: 'center',
                                  borderBottom: selectedChunk === idx ? '2px solid #049FD9' : 'none'
                                }}
                              >
                                <div>
                                  <strong style={{ color: '#049FD9' }}>Chunk {chunk.chunk_number}</strong>
                                  <span style={{ marginLeft: '10px', fontSize: '0.9rem', color: '#6c757d' }}>
                                    from: {chunk.source_id}
                                  </span>
                                  {chunk.score && (
                                    <span style={{ marginLeft: '10px', fontSize: '0.85rem', color: '#005073' }}>
                                      (Score: {chunk.score.toFixed(3)})
                                    </span>
                                  )}
                                </div>
                                <span>{selectedChunk === idx ? '▼' : '▶'}</span>
                              </div>
                              
                              {selectedChunk === idx && (
                                <div style={{ padding: '15px', background: '#fafafa' }}>
                                  <pre style={{
                                    background: '#1e1e1e',
                                    color: '#d4d4d4',
                                    padding: '15px',
                                    borderRadius: '4px',
                                    overflow: 'auto',
                                    fontSize: '0.85rem',
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    margin: 0
                                  }}>
                                    {chunk.text}
                                  </pre>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '20px' }}>
                  {references.length > 0 && (
                    <button
                      onClick={() => setShowReferences(!showReferences)}
                      className="btn btn-secondary"
                    >
                      {showReferences ? 'Hide' : 'Show'} Context References ({references.length})
                    </button>
                  )}
                  
                  {fullPrompt && (
                    <button
                      onClick={() => setShowPrompt(!showPrompt)}
                      className="btn btn-secondary"
                    >
                      {showPrompt ? 'Hide' : 'Show'} Full Prompt
                    </button>
                  )}
                </div>
                
                {showReferences && references.length > 0 && (
                  <div className="references-section">
                    <h4 style={{ color: '#049FD9', marginTop: '20px', marginBottom: '10px' }}>Context References:</h4>
                    <div className="references-list">
                      {references.map((ref, idx) => (
                        <div key={idx} className="reference-item">
                          <div className="reference-header">
                            <span className="reference-source">{ref.source_id}</span>
                            <span className="reference-score">Score: {ref.score?.toFixed(3) || 'N/A'}</span>
                          </div>
                          <div className="reference-text">{ref.text_preview}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                {showPrompt && fullPrompt && (
                  <div className="references-section">
                    <h4 style={{ color: '#049FD9', marginTop: '20px', marginBottom: '10px' }}>Full Prompt Sent to LLM:</h4>
                    <pre className="code-block" style={{ background: '#1e1e1e', color: '#d4d4d4', padding: '20px', borderRadius: '8px', overflow: 'auto', fontSize: '0.85rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {fullPrompt}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}

export default App;

