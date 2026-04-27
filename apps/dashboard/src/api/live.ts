import type { LiveDebugEntry, LiveDetection, ManualEntryPayload } from "../types";

import apiClient from "./client";

export async function getLiveRecent(limit = 100): Promise<LiveDetection[]> {
  const { data } = await apiClient.get<LiveDetection[]>("/live/recent", {
    params: { limit },
  });
  return data;
}

export async function getLiveDebug(limit = 150): Promise<LiveDebugEntry[]> {
  const { data } = await apiClient.get<LiveDebugEntry[]>("/live/debug", {
    params: { limit },
  });
  return data;
}

export async function adjustDetection(
  eventId: string,
  payload: { swap_type: string; notes?: string | null },
): Promise<void> {
  await apiClient.post(`/live/detections/${eventId}/adjust`, {
    swap_type: payload.swap_type,
    notes: payload.notes ?? null,
  });
}

export async function submitManualDetection(
  payload: ManualEntryPayload,
): Promise<{ status: string; id: string; row: LiveDetection }> {
  const { data } = await apiClient.post("/live/manual-detection", {
    plate_text: payload.plate_text || undefined,
    route_number: payload.route_number || undefined,
    camera_id: payload.camera_id ?? "manual",
    camera_name: payload.camera_name ?? "Manual entry",
    notes: payload.notes ?? null,
    confidence: payload.confidence ?? 1.0,
  });
  return data;
}
