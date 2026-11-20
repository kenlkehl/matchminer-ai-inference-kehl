import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { api, type LLMStatus, type ModelList } from "@/lib/api";
import { llmOptions } from "@/lib/queries";

const SELECT = "h-9 rounded-md border border-input bg-background px-2 text-sm";

/**
 * The LLM checks' endpoint (the guideline fetcher's Model Configuration, reduced): an
 * OpenAI-compatible URL whose models are read from its /v1/models, or a Vertex profile. The choice
 * holds for the server's session; the command line (--llm-endpoint, --llm-model) sets its start.
 */
export function LLMPanel() {
  const status = useQuery(llmOptions());
  if (!status.data) return status.isError ? <p>{status.error.message}</p> : <Spinner />;
  return <LLMForm status={status.data} />;
}

function LLMForm({ status }: { status: LLMStatus }) {
  const client = useQueryClient();
  const c = status.connection;
  const [backend, setBackend] = useState(c.backend);
  const [endpoint, setEndpoint] = useState(c.vertex ? status.defaults.endpoint : c.endpoint);
  const [model, setModel] = useState(c.vertex ? status.defaults.model : c.model);
  const [models, setModels] = useState<ModelList | null>(null);
  useEffect(() => {
    setModels(null);
  }, [backend, endpoint]);
  const vertex = status.vertex_profiles[backend];
  const check = useMutation({
    mutationFn: () => api.post<ModelList>("/api/llm/models", { backend, endpoint }),
    onSuccess: (found) => {
      setModels(found);
      const first = found.models.at(0);
      if (first && !found.models.some((m) => m.id === model)) setModel(first.id);
    },
  });
  const save = useMutation({
    mutationFn: () => api.post<LLMStatus>("/api/llm", { backend, endpoint, model }),
    onSuccess: (next) => {
      client.setQueryData(["llm"], next);
    },
  });
  return (
    <div className="flex flex-col gap-3 text-sm">
      <p className="rounded-md bg-amber-50 px-3 py-2 text-amber-900">{status.notice}</p>
      <p>
        <span className="text-gray-500">In use:</span>{" "}
        {c.configured ? <strong>{c.label}</strong> : <strong>none (the LLM checks are off)</strong>}
      </p>
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-gray-500">Connection</span>
          <select
            className={SELECT}
            value={backend}
            onChange={(e) => {
              setBackend(e.target.value);
            }}
          >
            <option value="openai">OpenAI-compatible server (vLLM)</option>
            {Object.entries(status.vertex_profiles).map(([key, p]) => (
              <option key={key} value={key}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        {vertex ? (
          <p className="pb-2 text-xs text-gray-500">
            {vertex.model} in project {vertex.project}, with this server's Google Application
            Default Credentials
          </p>
        ) : (
          <>
            <label className="flex min-w-72 flex-1 flex-col gap-1">
              <span className="text-xs text-gray-500">Endpoint (/v1 base URL)</span>
              <Input
                value={endpoint}
                onChange={(e) => {
                  setEndpoint(e.target.value);
                }}
                placeholder={status.defaults.endpoint}
              />
            </label>
            <label className="flex min-w-72 flex-1 flex-col gap-1">
              <span className="text-xs text-gray-500">Model</span>
              {models?.models.length ? (
                <select
                  className={SELECT}
                  value={model}
                  onChange={(e) => {
                    setModel(e.target.value);
                  }}
                >
                  {models.models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                      {m.max_model_len ? ` (${m.max_model_len.toLocaleString()} tokens)` : ""}
                    </option>
                  ))}
                </select>
              ) : (
                <Input
                  value={model}
                  onChange={(e) => {
                    setModel(e.target.value);
                  }}
                />
              )}
            </label>
          </>
        )}
        <Button
          variant="outline"
          disabled={check.isPending}
          onClick={() => {
            check.mutate();
          }}
        >
          {check.isPending ? <Spinner /> : null}
          Check endpoint
        </Button>
        <Button
          disabled={save.isPending}
          onClick={() => {
            save.mutate();
          }}
        >
          Use this endpoint
        </Button>
      </div>
      {models ? (
        <p className={models.available ? "text-green-700" : "text-red-700"}>
          {models.available
            ? `Reachable: ${String(models.models.length)} ${models.models.length === 1 ? "model" : "models"} at ${models.endpoint}`
            : models.error}
        </p>
      ) : null}
      {check.isError ? <p className="text-red-700">{check.error.message}</p> : null}
      {save.isError ? <p className="text-red-700">{save.error.message}</p> : null}
      <dl className="flex flex-col gap-0.5 text-xs text-gray-600">
        <div>
          <dt className="inline font-medium">Good option check: </dt>
          <dd className="inline">
            {status.good_option.available ? "available" : status.good_option.reason}
          </dd>
        </div>
        <div>
          <dt className="inline font-medium">Per-criterion LLM check: </dt>
          <dd className="inline">
            {status.criteria.available ? "available" : status.criteria.reason}
          </dd>
        </div>
        <div>
          <dt className="inline font-medium">Inference package: </dt>
          <dd className="inline">
            {status.package.version ?? "?"} ({status.package.path})
          </dd>
        </div>
      </dl>
    </div>
  );
}
