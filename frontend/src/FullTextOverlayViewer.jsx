import React, { useState } from 'react';

const FullTextOverlayViewer = ({ fullText, plagiarismMatches, aiProbability }) => {
  const [activeTab, setActiveTab] = useState('plagiarism'); // 'plagiarism' or 'ai'

  if (!fullText) return null;

  // Helper to escape regex special characters
  const escapeRegExp = (string) => {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  };

  const renderHighlightedText = () => {
    let content = fullText;

    if (activeTab === 'plagiarism' && plagiarismMatches) {
      // Sort matches by length descending to avoid partial replacements within longer matches
      const sortedMatches = [...plagiarismMatches].sort((a, b) => b.matched_text.length - a.matched_text.length);
      
      sortedMatches.forEach((match, index) => {
        if (!match.matched_text || match.matched_text.length < 10) return;
        
        const colorClass = match.match_type === 'cited' ? 'bg-yellow-200 border-yellow-400' : 'bg-red-200 border-red-400';
        const label = match.match_type === 'cited' ? 'Cited' : 'Uncited';
        
        const escapedText = escapeRegExp(match.matched_text);
        const regex = new RegExp(`(${escapedText})`, 'gi');
        
        // Use a placeholder to avoid nested replacements
        content = content.replace(regex, `<mark class="px-1 rounded border-b-2 ${colorClass} cursor-help group relative" title="${match.title} (${match.similarity}%)">$1<span class="hidden group-hover:block absolute bottom-full left-0 mb-2 p-2 bg-slate-800 text-white text-xs rounded shadow-lg z-50 w-64">${label}: ${match.title}</span></mark>`);
      });
    } else if (activeTab === 'ai' && aiProbability > 20) {
      // For AI, we highlight the whole text if probability is high, 
      // or in a real implementation, we'd highlight specific sentences from the backend.
      // Here we simulate sentence-level highlighting for demo purposes if probability > 50%
      if (aiProbability > 50) {
        const sentences = content.split(/(?<=[.!?])\s+/);
        content = sentences.map(s => `<mark class="bg-cyan-100 border-b-2 border-cyan-300">${s}</mark>`).join(' ');
      }
    }

    return <div className="prose prose-slate max-w-none text-slate-700 leading-relaxed whitespace-pre-wrap font-serif" dangerouslySetInnerHTML={{ __html: content }} />;
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden mt-8">
      <div className="border-b border-slate-200 bg-slate-50 px-6 py-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
          <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Full-Text Integrity Viewer
        </h3>
        <div className="flex bg-slate-200 p-1 rounded-lg">
          <button 
            onClick={() => setActiveTab('plagiarism')}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'plagiarism' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-600 hover:text-slate-800'}`}
          >
            Similarity Overlay
          </button>
          <button 
            onClick={() => setActiveTab('ai')}
            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'ai' ? 'bg-white text-cyan-600 shadow-sm' : 'text-slate-600 hover:text-slate-800'}`}
          >
            AI Overlay
          </button>
        </div>
      </div>
      
      <div className="p-8 max-h-[600px] overflow-y-auto bg-slate-50/30">
        <div className="max-w-3xl mx-auto bg-white p-10 shadow-inner border border-slate-100 rounded-sm">
          {renderHighlightedText()}
        </div>
      </div>
      
      <div className="px-6 py-3 bg-slate-50 border-t border-slate-200 flex gap-6 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-red-200 border border-red-400 rounded"></span>
          <span>Uncited Similarity</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-yellow-200 border border-yellow-400 rounded"></span>
          <span>Cited Similarity</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-cyan-100 border border-cyan-300 rounded"></span>
          <span>AI-Likely Content</span>
        </div>
      </div>
    </div>
  );
};

export default FullTextOverlayViewer;
