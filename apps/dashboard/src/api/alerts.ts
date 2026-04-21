import apiClient from "./client";
import type {
  Alert,
  AlertFilters,
  PaginatedResponse,
  ResolveAlertPayload,
} from "../types";

export async function getAlerts(
  filters?: AlertFilters,
): Promise<PaginatedResponse<Alert>> {
  const { data } = await apiClient.get<PaginatedResponse<Alert>>("/alerts", {
    params: filters,
  });
  return data;
}

export async function resolveAlert(
  id: string,
  payload: ResolveAlertPayload,
): Promise<Alert> {
  const { data } = await apiClient.post<Alert>(
    `/alerts/${id}/resolve`,
    payload,
  );
  return data;
}
