
import React, { useState, useRef } from 'react';
import { Upload, FileSpreadsheet, Clipboard, Check, ChevronDown, FileText } from 'lucide-react';
import * as XLSX from 'xlsx';
import { Dataset } from '../types';

interface Props {
  onDataLoaded: (dataset: Dataset) => void;
}

const DataIngestion: React.FC<Props> = ({ onDataLoaded }) => {
  const [activeTab, setActiveTab] = useState<'file' | 'paste'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [workbook, setWorkbook] = useState<XLSX.WorkBook | null>(null);
  const [sheetNames, setSheetNames] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string>('');
  const [pasteContent, setPasteContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      processFile(selectedFile);
    }
  };

  const processFile = (file: File) => {
    setFile(file);
    setIsLoading(true);
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const bstr = evt.target?.result;
        const wb = XLSX.read(bstr, { type: 'binary' });
        setWorkbook(wb);
        setSheetNames(wb.SheetNames);
        if (wb.SheetNames.length > 0) {
          setSelectedSheet(wb.SheetNames[0]);
        }
      } catch (err) {
        console.error("Error reading file:", err);
        alert("Failed to read Excel file.");
      } finally {
        setIsLoading(false);
      }
    };
    reader.readAsBinaryString(file);
  };

  const handleConfirmImport = () => {
    if (activeTab === 'file') {
      if (!workbook || !selectedSheet) return;
      
      const ws = workbook.Sheets[selectedSheet];
      const data = XLSX.utils.sheet_to_json<any>(ws);
      
      if (data.length === 0) {
        alert("Selected sheet is empty.");
        return;
      }

      const headers = Object.keys(data[0]);
      onDataLoaded({
        fileName: `${file?.name || 'Data'} - ${selectedSheet}`,
        headers,
        rows: data
      });

    } else {
      // Handle Paste
      if (!pasteContent.trim()) return;
      
      try {
        // Simple TSV/CSV parser for pasted content
        const rows = pasteContent.trim().split('\n').map(row => row.split('\t'));
        if (rows.length < 2) throw new Error("Not enough data");
        
        const headers = rows[0].map(h => h.trim());
        const data = rows.slice(1).map(row => {
          const obj: any = {};
          headers.forEach((h, i) => {
            // Try to convert to number if possible
            const val = row[i]?.trim();
            obj[h] = isNaN(Number(val)) ? val : Number(val);
          });
          return obj;
        });

        onDataLoaded({
          fileName: 'Pasted Data',
          headers,
          rows: data
        });
      } catch (e) {
        alert("Could not parse pasted data. Ensure it is copied from Excel (tab-separated).");
      }
    }
  };
  const handleLoadExample = async () => {
    setIsLoading(true);
    try {
        const response = await fetch('/Example.xlsx');
        if (!response.ok) throw new Error("Example file not found");
        
        const blob = await response.blob();
        const reader = new FileReader();
        
        reader.onload = (evt) => {
            try {
                const bstr = evt.target?.result;
                const wb = XLSX.read(bstr, { type: 'binary' });
                
                // Set file metadata as if user uploaded it
                setFile(new File([blob], "Example.xlsx", { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }));
                setWorkbook(wb);
                setSheetNames(wb.SheetNames);
                if (wb.SheetNames.length > 0) {
                    setSelectedSheet(wb.SheetNames[0]);
                }
                setActiveTab('file');
            } catch (err) {
                console.error("Error parsing example file:", err);
                alert("Failed to parse example file.");
            } finally {
                setIsLoading(false);
            }
        };
        reader.readAsBinaryString(blob);
        
    } catch (e) {
        console.error(e);
        alert("Failed to load example data.");
        setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-indigo-500">
      <div className="p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <Upload className="text-indigo-600" size={24} />
              Upload Data
            </h2>
          </div>
          <button 
            onClick={handleLoadExample}
            disabled={isLoading}
            className="bg-indigo-50 text-indigo-600 px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 hover:bg-indigo-100 transition-colors disabled:opacity-50"
          >
            <FileSpreadsheet size={16} className="text-yellow-500" />
            {isLoading ? "Loading..." : "Load Example Data"}
          </button>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 bg-slate-100 p-1 rounded-lg w-fit mb-6">
          <button
            onClick={() => setActiveTab('file')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'file' 
                ? 'bg-white text-indigo-600 shadow-sm' 
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            Excel File Upload
          </button>
          <button
            onClick={() => setActiveTab('paste')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === 'paste' 
                ? 'bg-white text-indigo-600 shadow-sm' 
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            Copy & Paste
          </button>
        </div>

        <p className="text-sm text-slate-500 mb-4">
          Supports .xlsx, .xls, .csv formats, or copy directly from Excel.
        </p>

        {/* Content Area */}
        <div className="transition-all duration-300">
          {activeTab === 'file' ? (
            <div 
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors bg-slate-50/50 ${
                file ? 'border-indigo-300 bg-indigo-50/30' : 'border-slate-300 hover:border-indigo-400'
              }`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const droppedFile = e.dataTransfer.files[0];
                if(droppedFile) processFile(droppedFile);
              }}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                accept=".xlsx,.xls,.csv" 
                onChange={handleFileChange} 
              />
              
              {!file ? (
                <div 
                  className="cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div className="w-14 h-14 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Upload size={28} />
                  </div>
                  <p className="text-lg font-medium text-slate-700 mb-1">Click or Drag file here</p>
                  <p className="text-sm text-slate-400">No file selected</p>
                </div>
              ) : (
                <div>
                  <div className="w-14 h-14 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <FileText size={28} />
                  </div>
                  <p className="text-lg font-medium text-slate-700 mb-1">{file.name}</p>
                  <p className="text-sm text-slate-500 mb-4">{(file.size / 1024).toFixed(1)} KB</p>
                  <button 
                    onClick={() => { setFile(null); setWorkbook(null); }}
                    className="text-sm text-red-500 hover:text-red-700 font-medium underline"
                  >
                    Change File
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Paste Excel Data (Header row required)
              </label>
              <textarea 
                className="w-full h-40 p-3 rounded-lg border-slate-300 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 font-mono text-sm"
                placeholder={`Date\tSales\n2024-01-01\t100\n2024-01-02\t150`}
                value={pasteContent}
                onChange={(e) => setPasteContent(e.target.value)}
              />
            </div>
          )}
        </div>

        {/* Worksheet Selector (Only for File mode with workbook loaded) */}
        {activeTab === 'file' && workbook && (
          <div className="mt-6 bg-indigo-50 border border-indigo-100 rounded-xl p-4 flex flex-col md:flex-row items-end gap-4 animate-fade-in-up">
            <div className="flex-1 w-full">
              <label className="block text-sm font-medium text-indigo-900 mb-1">
                Select Worksheet
              </label>
              <div className="relative">
                <select 
                  value={selectedSheet}
                  onChange={(e) => setSelectedSheet(e.target.value)}
                  className="w-full appearance-none bg-white border border-indigo-200 text-slate-700 py-2.5 px-4 pr-8 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                >
                  {sheetNames.map((name, idx) => (
                    <option key={name} value={name}>{idx + 1}. {name}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-3 text-indigo-400 pointer-events-none" size={16} />
              </div>
            </div>
            <button 
              onClick={handleConfirmImport}
              className="w-full md:w-auto bg-indigo-600 text-white px-6 py-2.5 rounded-lg font-medium shadow hover:bg-indigo-700 transition-colors flex items-center justify-center gap-2"
            >
              <Check size={18} />
              Confirm Import
            </button>
          </div>
        )}

        {/* Paste Mode Confirm Button */}
        {activeTab === 'paste' && pasteContent && (
           <div className="mt-4 flex justify-end">
             <button 
              onClick={handleConfirmImport}
              className="bg-emerald-600 text-white px-6 py-2.5 rounded-lg font-medium shadow hover:bg-emerald-700 transition-colors flex items-center gap-2"
            >
              <Clipboard size={18} />
              Import Data
            </button>
           </div>
        )}

      </div>
    </div>
  );
};

export default DataIngestion;
