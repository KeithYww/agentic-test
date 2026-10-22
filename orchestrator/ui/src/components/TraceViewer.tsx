// SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
//
// SPDX-License-Identifier: AGPL-3.0-only

import { useEffect } from 'react';
import { X, GitBranch, User, Bot, Wrench, Loader2 } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { dashboardApi } from '../api/dashboardApi';
import type { TraceMessage, TracePart } from '../types/dashboard';

interface TraceViewerProps {
  isOpen: boolean;
  onClose: () => void;
  taskId: string;
  isRunning?: boolean;
}

function formatContent(content: TracePart['content']): string {
  if (content === undefined || content === null) return '';
  if (typeof content === 'string') return content;
  return content
    .map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
    .join('\n');
}

function PartCard({ part, role }: { part: TracePart; role: 'request' | 'response' }) {
  switch (part.part_kind) {
    case 'user-prompt':
      return (
        <div className="flex gap-3">
          <User className="w-4 h-4 text-sky-400 flex-shrink-0 mt-0.5" />
          <div className="text-slate-200 whitespace-pre-wrap break-all">{formatContent(part.content)}</div>
        </div>
      );
    case 'system-prompt':
      return (
        <div className="flex gap-3 text-slate-500">
          <GitBranch className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div className="whitespace-pre-wrap break-all italic">{formatContent(part.content)}</div>
        </div>
      );
    case 'tool-return':
      return (
        <div className="flex gap-3">
          <Wrench className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-amber-300 text-xs mb-1">Tool result: {part.tool_name}</div>
            <div className="text-slate-300 whitespace-pre-wrap break-all">{formatContent(part.content)}</div>
          </div>
        </div>
      );
    case 'tool-call':
      return (
        <div className="flex gap-3">
          <Wrench className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="text-amber-300 text-xs mb-1">Tool call: {part.tool_name}</div>
            <pre className="text-slate-300 whitespace-pre-wrap break-all text-xs bg-slate-800/50 rounded p-2">
              {typeof part.args === 'string' ? part.args : JSON.stringify(part.args, null, 2)}
            </pre>
          </div>
        </div>
      );
    case 'text':
      return (
        <div className="flex gap-3">
          <Bot className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />
          <div className="text-slate-200 whitespace-pre-wrap break-all">{formatContent(part.content)}</div>
        </div>
      );
    case 'thinking':
      return (
        <div className="flex gap-3 text-indigo-300/80">
          <Bot className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div className="whitespace-pre-wrap break-all italic">{formatContent(part.content)}</div>
        </div>
      );
    default:
      return (
        <div className="flex gap-3 text-slate-500">
          <span className="flex-shrink-0">[{part.part_kind || role}]</span>
          <div className="whitespace-pre-wrap break-all">{formatContent(part.content)}</div>
        </div>
      );
  }
}

export function TraceViewer({ isOpen, onClose, taskId, isRunning }: TraceViewerProps) {
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  const { data: trace, isLoading } = useQuery({
    queryKey: ['trace', taskId],
    queryFn: () => dashboardApi.getTaskTrace(taskId),
    enabled: isOpen,
    // The trace artifact is only emitted once the task reaches a terminal state, so
    // keep polling while the task is still running and no trace is available yet.
    refetchInterval: (query) => (isRunning && !query.state.data ? 2000 : false),
  });

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-slate-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col border border-slate-700">
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-slate-700/50 rounded-lg">
              <GitBranch className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Execution Trace</h2>
              <span className="text-xs text-slate-400">Task: {taskId.substring(0, 8)}...</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 bg-slate-900 text-xs space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-full text-slate-500 animate-pulse">
              Loading trace...
            </div>
          ) : !trace ? (
            <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-2">
              {isRunning ? (
                <>
                  <Loader2 className="w-8 h-8 opacity-50 animate-spin" />
                  <p>Waiting for the task to complete...</p>
                </>
              ) : (
                <>
                  <GitBranch className="w-8 h-8 opacity-50" />
                  <p>No trace available for this task</p>
                </>
              )}
            </div>
          ) : (
            (trace as TraceMessage[]).map((message, mIndex) => (
              <div key={mIndex} className="space-y-2">
                {message.parts.map((part, pIndex) => (
                  <PartCard key={pIndex} part={part} role={message.kind === 'response' ? 'response' : 'request'} />
                ))}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
