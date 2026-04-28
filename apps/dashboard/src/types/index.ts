/* ------------------------------------------------------------------ */
/*  Enums                                                             */
/* ------------------------------------------------------------------ */

export enum GateType {
  ENTRY = "entry",
  EXIT = "exit",
}

export enum TripStatus {
  OPEN = "open",
  CLOSED = "closed",
  OVERDUE = "overdue",
}

export enum AnomalyCode {
  NONE = "none",
  LOW_CONFIDENCE = "low_confidence",
  PLATE_MISMATCH = "plate_mismatch",
  DUPLICATE_EVENT = "duplicate_event",
  ORPHAN_ENTRY = "orphan_entry",
  RAPID_RE_ENTRY = "rapid_re_entry",
}

export enum AlertSeverity {
  INFO = "info",
  WARNING = "warning",
  CRITICAL = "critical",
}

export enum ReviewStatus {
  PENDING = "pending",
  APPROVED = "approved",
  CORRECTED = "corrected",
  REJECTED = "rejected",
}

export enum CameraStatus {
  ONLINE = "online",
  OFFLINE = "offline",
  ERROR = "error",
}

export enum BusStatus {
  INSIDE = "inside",
  OUTSIDE = "outside",
  UNKNOWN = "unknown",
}

/* ------------------------------------------------------------------ */
/*  Domain models                                                     */
/* ------------------------------------------------------------------ */

export interface Camera {
  id: string;
  name: string;
  gate_type: GateType;
  stream_url: string;
  status: CameraStatus;
  /** When false, vision worker skips this camera (configure in Cameras page). */
  is_active: boolean;
  last_heartbeat: string | null;
  created_at: string;
  updated_at: string;
}

/** Result of POST /cameras/:id/probe (TCP check from API server). */
export interface CameraProbeResult {
  camera_id: string;
  tcp_reachable: boolean;
  status: CameraStatus;
  hint: string | null;
}

export interface GateEvent {
  id: string;
  camera_id: string;
  camera_name: string;
  gate_type: GateType;
  plate_number: string;
  confidence: number;
  snapshot_url: string | null;
  raw_candidates: string[];
  anomaly_code: AnomalyCode;
  review_status: ReviewStatus;
  trip_id: string | null;
  route_number?: string | null;
  timestamp: string;
  created_at: string;
}

export interface Trip {
  id: string;
  plate_number: string;
  exit_event_id: string;
  entry_event_id: string | null;
  exit_time: string;
  entry_time: string | null;
  duration_seconds: number | null;
  status: TripStatus;
  anomaly_code: AnomalyCode;
  created_at: string;
  updated_at: string;
}

export interface Alert {
  id: string;
  trip_id: string | null;
  event_id: string | null;
  plate_number: string;
  severity: AlertSeverity;
  alert_type: string;
  message: string;
  resolved: boolean;
  resolved_at: string | null;
  resolved_by: string | null;
  resolution_note: string | null;
  created_at: string;
}

export interface PlateDetail {
  plate_number: string;
  current_status: BusStatus;
  last_seen: string | null;
  last_camera: string | null;
  total_trips: number;
  recent_trips: Trip[];
  recent_events: GateEvent[];
}

export interface PlateStatistics {
  plate_number: string;
  total_trips: number;
  avg_duration_seconds: number;
  max_duration_seconds: number;
  min_duration_seconds: number;
  total_anomalies: number;
  overdue_count: number;
  first_seen: string;
  last_seen: string;
}

export interface ManualCorrection {
  id: string;
  event_id: string;
  original_plate: string;
  corrected_plate: string;
  reason: string;
  corrected_by: string;
  created_at: string;
}

/* ------------------------------------------------------------------ */
/*  Dashboard summary                                                 */
/* ------------------------------------------------------------------ */

export interface DashboardSummary {
  total_buses_known: number;
  buses_outside_now: number;
  buses_overdue: number;
  alerts_today: number;
  events_today: number;
  trips_today: number;
  recent_events: GateEvent[];
  active_alerts: Alert[];
}

/** Row from GET /live/recent or WebSocket `detection` payload. */
export interface LiveDetection {
  id: string;
  type: string;
  plate_text: string;
  confidence: number;
  camera_id: string;
  camera_name: string;
  snapshot_base64: string | null;
  detected_at: string;
  /** Set when the plate is registered in the vehicle registry. */
  route_number?: string;
  route_name?: string;
  driver_name?: string;
  is_registered?: boolean;
  /** Route OCR'd from the bus placard / LED display (vision worker reads "AR-29" from windshield). */
  detected_route?: string | null;
  /** True when detected_route ≠ registered route_number — bus may have been swapped. */
  is_mismatch?: boolean;
  /** Set after staff resolve the mismatch via the Adjust button. */
  swap_type?: string | null;
  swap_notes?: string | null;
  swap_resolved?: boolean;
  swap_resolved_by?: string | null;
  /**
   * True when only the route placard was visible — the camera couldn't OCR
   * the plate. The registry suggestion (if any) lives in `suggested_plate`.
   * UI renders a yellow triangle and a "Registry suggests" hint, never
   * pretends the plate was read.
   */
  plate_from_storage?: boolean;
  /**
   * Registry plate associated with the detected route, surfaced as a HINT
   * when `plate_text` is empty. Populated by the API on route-only posts.
   */
  suggested_plate?: string | null;
  /** Plate was actually read by the camera. */
  has_plate?: boolean;
  /** Route was actually read from the placard / LED display. */
  has_route?: boolean;
  /** "vision" = camera worker; "manual" = operator entry from the dashboard. */
  source?: "vision" | "manual" | string;
  /** Operator notes on a manual entry. */
  notes?: string | null;
}

export interface ManualEntryPayload {
  plate_text?: string;
  route_number?: string;
  camera_id?: string;
  camera_name?: string;
  notes?: string;
  confidence?: number;
}

/** Structured pipeline / ingest debug (GET /live/debug, WebSocket `debug`). */
export interface LiveDebugEntry {
  id: string;
  at: string;
  message: string;
  source: string;
  detail: Record<string, unknown>;
}

/* ------------------------------------------------------------------ */
/*  Vehicles                                                          */
/* ------------------------------------------------------------------ */

export interface Vehicle {
  id: string;
  plate_number: string;
  vehicle_type: string;
  route_number: string;
  route_name: string;
  driver_name: string;
  driver_phone: string;
  capacity: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface CreateVehiclePayload {
  plate_number: string;
  vehicle_type?: string;
  route_number: string;
  route_name?: string;
  driver_name?: string;
  driver_phone?: string;
  capacity?: number;
}

export interface UpdateVehiclePayload {
  plate_number?: string;
  vehicle_type?: string;
  route_number?: string;
  route_name?: string;
  driver_name?: string;
  driver_phone?: string;
  capacity?: number;
  is_active?: boolean;
}

export interface RouteInfo {
  route_number: string;
  route_name: string;
  vehicle_count: number;
  vehicles: Vehicle[];
}

/* ------------------------------------------------------------------ */
/*  Users                                                             */
/* ------------------------------------------------------------------ */

export interface User {
  id: string;
  username: string;
  display_name: string;
  role: string;
  is_active: boolean;
  created_at: string;
  last_login: string | null;
}

export interface CreateUserPayload {
  username: string;
  password: string;
  display_name: string;
  role?: string;
}

export interface UpdateUserPayload {
  display_name?: string;
  role?: string;
  is_active?: boolean;
  password?: string;
}

/* ------------------------------------------------------------------ */
/*  API pagination / filter helpers                                   */
/* ------------------------------------------------------------------ */

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

export interface EventFilters {
  plate_number?: string;
  gate_type?: GateType;
  camera_id?: string;
  from_date?: string;
  to_date?: string;
  review_status?: ReviewStatus;
  page?: number;
  page_size?: number;
}

export interface TripFilters {
  plate_number?: string;
  status?: TripStatus;
  from_date?: string;
  to_date?: string;
  anomaly_code?: AnomalyCode;
  page?: number;
  page_size?: number;
}

export interface AlertFilters {
  severity?: AlertSeverity;
  resolved?: boolean;
  plate_number?: string;
  page?: number;
  page_size?: number;
}

export interface CorrectionFilters {
  event_id?: string;
  page?: number;
  page_size?: number;
}

export interface CreateCameraPayload {
  name: string;
  gate_type: GateType;
  stream_url: string;
}

export interface UpdateCameraPayload {
  name?: string;
  gate_type?: GateType;
  stream_url?: string;
  status?: CameraStatus;
  is_active?: boolean;
}

export interface CreateCorrectionPayload {
  event_id: string;
  corrected_plate: string;
  reason: string;
  corrected_by: string;
}

export interface ResolveAlertPayload {
  resolved_by: string;
  resolution_note: string;
}
