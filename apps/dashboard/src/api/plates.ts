import apiClient from "./client";
import type { PlateDetail } from "../types";

export async function getPlateDetail(plateNumber: string): Promise<PlateDetail> {
  const { data } = await apiClient.get<PlateDetail>(
    `/plates/${encodeURIComponent(plateNumber)}`,
  );
  return data;
}
