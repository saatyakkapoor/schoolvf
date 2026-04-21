import apiClient from "./client";
import type {
  Camera,
  CameraProbeResult,
  CreateCameraPayload,
  UpdateCameraPayload,
} from "../types";

export async function getCameras(): Promise<Camera[]> {
  const { data } = await apiClient.get<Camera[]>("/cameras");
  return data;
}

export async function getCamera(id: string): Promise<Camera> {
  const { data } = await apiClient.get<Camera>(`/cameras/${id}`);
  return data;
}

export async function createCamera(
  payload: CreateCameraPayload,
): Promise<Camera> {
  const { data } = await apiClient.post<Camera>("/cameras", payload);
  return data;
}

export async function updateCamera(
  id: string,
  payload: UpdateCameraPayload,
): Promise<Camera> {
  const { data } = await apiClient.patch<Camera>(`/cameras/${id}`, payload);
  return data;
}

/** TCP probe from API container/host — not a full RTSP auth test. */
export async function probeCamera(id: string): Promise<CameraProbeResult> {
  const { data } = await apiClient.post<CameraProbeResult>(`/cameras/${id}/probe`);
  return data;
}
