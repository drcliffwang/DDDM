
import React from 'react';
import { Database } from 'lucide-react';
import { Dataset } from '../types';

interface Props {
  dataset: Dataset;
}

const DataPreview: React.FC<Props> = ({ dataset }) => {
  // Only show top 50 records
  const previewRows = dataset.rows.slice(0, 50);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden border-l-4 border-l-teal-500">
      <div className="p-8">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
              <Database className="text-teal-600" size={24} />
              資料預覽
            </h2>
            <p className="mt-1 text-slate-500">Showing first {previewRows.length} rows of {dataset.rows.length} total records</p>
          </div>
        </div>

        <div className="overflow-auto max-h-[400px] border border-slate-200 rounded-lg shadow-inner custom-scrollbar">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50 sticky top-0 z-10">
              <tr>
                {dataset.headers.map((header) => (
                  <th
                    key={header}
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-bold text-slate-500 uppercase tracking-wider whitespace-nowrap bg-slate-50"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-slate-200">
              {previewRows.map((row, rowIndex) => (
                <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-slate-50/50 hover:bg-slate-100 transition-colors'}>
                  {dataset.headers.map((header, colIndex) => (
                    <td
                      key={`${rowIndex}-${colIndex}`}
                      className="px-6 py-3 whitespace-nowrap text-sm text-slate-700"
                    >
                      {row[header] !== null && row[header] !== undefined 
                        ? String(row[header]) 
                        : <span className="text-slate-300 italic">null</span>}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        <div className="mt-2 text-right">
           <span className="text-xs text-slate-400">
             * Only the first 50 rows are displayed for preview.
           </span>
        </div>
      </div>
    </div>
  );
};

export default DataPreview;
