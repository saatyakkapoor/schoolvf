import apiClient from "./client";
import type {
  ManualCorrection,
  CreateCorrectionPayload,
  CorrectionFilters,
  PaginatedResponse,
} from "../types";

export async function createCorrection(
  payload: CreateCorrectionPayload,
): Promise<ManualCorrection> {
  const { data } = await apiClient.post<ManualCorrection>(
    "/corrections",
    payload,
  );
  return data;
}

export async function getCorrections(
  filters?: CorrectionFilters,
): Promise<PaginatedResponse<ManualCorrection>> {
  const { data } = await apiClient.get<PaginatedResponse<ManualCorrection>>(
    "/corrections",
    { params: filters },
  );
  return data;
}
