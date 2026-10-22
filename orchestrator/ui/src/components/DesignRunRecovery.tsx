// SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
//
// SPDX-License-Identifier: AGPL-3.0-only

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { dashboardApi } from '../api/dashboardApi';

const recoveryMessages: Record<string, string> = {
  workflow_changed: '任务创建后模型或流程配置已变化，请联系维护人员核实后处理。',
  upgrade_manifest_changed: '任务恢复时的代码或环境已变化，请联系维护人员核实。',
  resume_preconditions_not_met: '任务当前状态、剩余额度、截止时间或未决操作不满足续跑条件，请联系维护人员核实。',
  stale_run_generation: '任务已被其他人操作，请刷新后查看最新状态。',
};

function errorMessage(error: unknown): string {
  const detail = isAxiosError(error) ? error.response?.data?.detail : null;
  return typeof detail === 'string'
    ? recoveryMessages[detail] || detail
    : '暂时无法处理请求，请稍后刷新重试。';
}

export function DesignRunRecovery({ runId }: { runId: string }) {
  const client = useQueryClient();
  const queryKey = ['design-run', runId];
  const { data: run, error, isLoading } = useQuery({
    queryKey,
    queryFn: () => dashboardApi.getDesignRun(runId),
  });
  const resume = useMutation({
    mutationFn: (generation: number) => dashboardApi.resumeDesignRun(runId, generation),
    onSettled: () => { void client.invalidateQueries({ queryKey }); },
  });

  return (
    <section className="bg-slate-800/80 rounded-xl p-6 border border-indigo-500/50 mb-6" aria-label="任务续跑">
      <h2 className="text-xl font-semibold mb-3">任务续跑</h2>
      {isLoading && <p>正在读取任务进度…</p>}
      {error && <p role="alert" className="text-red-300">{errorMessage(error)}</p>}
      {run && (
        <>
          <p className="text-slate-200">{run.title || run.story_id} · {run.state}</p>
          <p className="text-xs text-slate-400 mt-1">任务 ID：{run.run_id}</p>
          <p className="mt-3">已保存 {run.completed_units} 项结果；已用 {run.used_tokens.toLocaleString()} / {run.token_limit.toLocaleString()} tokens</p>
          {run.reason && <p className="text-amber-300 mt-2">暂停原因：{run.reason}</p>}
          {run.url?.startsWith('https://') && (
            <a className="text-indigo-300 underline block mt-2" href={run.url} target="_blank" rel="noopener noreferrer">查看已有用例表</a>
          )}
          <p className="text-slate-400 mt-3">确认后从已保存的断点继续，沿用原任务预算和截止时间。不会重新生成已完成的结果。</p>
          <p className="text-slate-400 text-sm">截止时间：{new Date(run.deadline * 1000).toLocaleString()}</p>
          {!run.can_resume && run.blocked_reason && (
            <p className="text-amber-300 mt-2">{recoveryMessages[run.blocked_reason] || run.blocked_reason}</p>
          )}
          <button className="mt-4 px-4 py-2 rounded bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={!run.can_resume || resume.isPending || resume.isSuccess}
            onClick={() => resume.mutate(run.generation)}>
            {resume.isPending ? '正在提交…' : '确认续跑'}
          </button>
          {resume.isSuccess && <p role="status" className="text-emerald-300 mt-2">已提交续跑，请查看任务进度。</p>}
          {resume.error && <p role="alert" className="text-red-300 mt-2">{errorMessage(resume.error)}</p>}
        </>
      )}
    </section>
  );
}
