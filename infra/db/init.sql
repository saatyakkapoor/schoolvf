-- SchoolVF: School Bus Entry-Exit Monitoring System
-- Database initialization script

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- Table: cameras
-- ============================================================
CREATE TABLE cameras (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            VARCHAR(255) NOT NULL,
    gate_type       VARCHAR(20)  NOT NULL CHECK (gate_type IN ('entry', 'exit')),
    stream_url      TEXT         NOT NULL,
    status          VARCHAR(20)  NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'error')),
    roi_config      JSONB,
    sampling_fps    REAL         NOT NULL DEFAULT 2.0,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cameras_gate_type ON cameras (gate_type);
CREATE INDEX idx_cameras_status    ON cameras (status);

-- ============================================================
-- Table: raw_plate_reads
-- ============================================================
CREATE TABLE raw_plate_reads (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id         UUID         NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    captured_at       TIMESTAMPTZ  NOT NULL,
    plate_text_raw    VARCHAR(20)  NOT NULL,
    plate_text_norm   VARCHAR(20)  NOT NULL,
    detector_conf     REAL         NOT NULL,
    ocr_conf          REAL         NOT NULL,
    snapshot_path     TEXT,
    processing_meta   JSONB,
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_raw_plate_reads_camera_id      ON raw_plate_reads (camera_id);
CREATE INDEX idx_raw_plate_reads_captured_at     ON raw_plate_reads (captured_at);
CREATE INDEX idx_raw_plate_reads_plate_norm      ON raw_plate_reads (plate_text_norm);
CREATE INDEX idx_raw_plate_reads_camera_captured ON raw_plate_reads (camera_id, captured_at);

-- ============================================================
-- Table: gate_events
-- ============================================================
CREATE TABLE gate_events (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id         UUID         NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    direction         VARCHAR(10)  NOT NULL CHECK (direction IN ('entry', 'exit')),
    detected_at       TIMESTAMPTZ  NOT NULL,
    plate_final       VARCHAR(20)  NOT NULL,
    confidence_final  REAL         NOT NULL,
    best_snapshot_path TEXT,
    source_read_ids   JSONB,
    dedupe_key        VARCHAR(100) NOT NULL,
    review_status     VARCHAR(20)  NOT NULL DEFAULT 'auto' CHECK (review_status IN ('auto', 'pending_review', 'confirmed', 'rejected')),
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_gate_events_dedupe_key      ON gate_events (dedupe_key);
CREATE INDEX idx_gate_events_camera_id              ON gate_events (camera_id);
CREATE INDEX idx_gate_events_direction              ON gate_events (direction);
CREATE INDEX idx_gate_events_detected_at            ON gate_events (detected_at);
CREATE INDEX idx_gate_events_plate_final            ON gate_events (plate_final);
CREATE INDEX idx_gate_events_plate_direction_time   ON gate_events (plate_final, direction, detected_at);

-- ============================================================
-- Table: trips
-- ============================================================
CREATE TABLE trips (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plate_number         VARCHAR(20)  NOT NULL,
    exit_event_id        UUID         NOT NULL REFERENCES gate_events(id) ON DELETE CASCADE,
    entry_event_id       UUID         REFERENCES gate_events(id) ON DELETE SET NULL,
    exit_time            TIMESTAMPTZ  NOT NULL,
    entry_time           TIMESTAMPTZ,
    duration_seconds     INTEGER,
    status               VARCHAR(20)  NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed', 'anomaly')),
    anomaly_code         VARCHAR(50),
    manually_corrected   BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trips_plate_number    ON trips (plate_number);
CREATE INDEX idx_trips_status          ON trips (status);
CREATE INDEX idx_trips_exit_time       ON trips (exit_time);
CREATE INDEX idx_trips_entry_time      ON trips (entry_time);
CREATE INDEX idx_trips_exit_event_id   ON trips (exit_event_id);
CREATE INDEX idx_trips_entry_event_id  ON trips (entry_event_id);
CREATE INDEX idx_trips_plate_status    ON trips (plate_number, status);
CREATE INDEX idx_trips_anomaly_code    ON trips (anomaly_code) WHERE anomaly_code IS NOT NULL;

-- ============================================================
-- Table: plate_statistics
-- ============================================================
CREATE TABLE plate_statistics (
    plate_number          VARCHAR(20) PRIMARY KEY,
    trip_count            INTEGER      NOT NULL DEFAULT 0,
    avg_duration_seconds  REAL,
    min_duration_seconds  REAL,
    max_duration_seconds  REAL,
    last_seen_at          TIMESTAMPTZ,
    last_status           VARCHAR(20),
    updated_at            TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_plate_statistics_last_seen ON plate_statistics (last_seen_at);

-- ============================================================
-- Table: alerts
-- ============================================================
CREATE TABLE alerts (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trip_id          UUID         REFERENCES trips(id) ON DELETE SET NULL,
    plate_number     VARCHAR(20)  NOT NULL,
    alert_type       VARCHAR(50)  NOT NULL,
    severity         VARCHAR(20)  NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    resolved_at      TIMESTAMPTZ,
    resolution_note  TEXT
);

CREATE INDEX idx_alerts_trip_id       ON alerts (trip_id);
CREATE INDEX idx_alerts_plate_number  ON alerts (plate_number);
CREATE INDEX idx_alerts_alert_type    ON alerts (alert_type);
CREATE INDEX idx_alerts_severity      ON alerts (severity);
CREATE INDEX idx_alerts_created_at    ON alerts (created_at);
CREATE INDEX idx_alerts_unresolved    ON alerts (resolved_at) WHERE resolved_at IS NULL;

-- ============================================================
-- Table: manual_corrections
-- ============================================================
CREATE TABLE manual_corrections (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target_type    VARCHAR(50)  NOT NULL,
    target_id      UUID         NOT NULL,
    old_value      TEXT,
    new_value      TEXT,
    reason         TEXT,
    corrected_by   VARCHAR(255),
    corrected_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_manual_corrections_target      ON manual_corrections (target_type, target_id);
CREATE INDEX idx_manual_corrections_corrected_at ON manual_corrections (corrected_at);

-- ============================================================
-- Trigger: auto-update updated_at timestamps
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_cameras_updated_at
    BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_trips_updated_at
    BEFORE UPDATE ON trips
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_plate_statistics_updated_at
    BEFORE UPDATE ON plate_statistics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- App layer (dashboard auth, fleet, live ingest / trips) — used by FastAPI
-- ============================================================

CREATE TABLE app_users (
    id              VARCHAR(32) PRIMARY KEY,
    username        VARCHAR(128) NOT NULL UNIQUE,
    password_hash   VARCHAR(128) NOT NULL,
    display_name    VARCHAR(255) NOT NULL,
    role            VARCHAR(32)  NOT NULL,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL,
    last_login      TIMESTAMPTZ
);

CREATE TABLE app_vehicles (
    id              VARCHAR(32) PRIMARY KEY,
    plate_number    VARCHAR(32)  NOT NULL,
    vehicle_type    VARCHAR(32)  NOT NULL,
    route_number    VARCHAR(64)  NOT NULL,
    route_name      TEXT         NOT NULL DEFAULT '',
    driver_name     VARCHAR(255) NOT NULL DEFAULT '',
    driver_phone    VARCHAR(64)  NOT NULL DEFAULT '',
    capacity        INTEGER      NOT NULL DEFAULT 40,
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL,
    updated_at      TIMESTAMPTZ  NOT NULL
);
CREATE INDEX idx_app_vehicles_plate ON app_vehicles (plate_number);

CREATE TABLE app_gate_events (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    camera_id        VARCHAR(64)  NOT NULL,
    camera_name      VARCHAR(255) NOT NULL DEFAULT '',
    gate_type        VARCHAR(16)  NOT NULL CHECK (gate_type IN ('entry', 'exit')),
    direction        VARCHAR(16)  NOT NULL CHECK (direction IN ('entry', 'exit')),
    plate_number     VARCHAR(32)  NOT NULL,
    confidence       REAL         NOT NULL,
    snapshot_base64  TEXT,
    detected_at      TIMESTAMPTZ  NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_app_gate_events_plate ON app_gate_events (plate_number);
CREATE INDEX idx_app_gate_events_detected ON app_gate_events (detected_at DESC);

CREATE TABLE app_trips (
    id                 VARCHAR(32) PRIMARY KEY,
    plate_number       VARCHAR(32)  NOT NULL,
    exit_event_id      UUID         NOT NULL REFERENCES app_gate_events (id) ON DELETE CASCADE,
    entry_event_id     UUID         REFERENCES app_gate_events (id) ON DELETE SET NULL,
    exit_time          TIMESTAMPTZ  NOT NULL,
    entry_time         TIMESTAMPTZ,
    duration_seconds   INTEGER,
    status             VARCHAR(32)  NOT NULL CHECK (status IN ('open', 'closed', 'overdue')),
    anomaly_code       VARCHAR(64)  NOT NULL DEFAULT 'none',
    created_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_app_trips_plate ON app_trips (plate_number);
CREATE INDEX idx_app_trips_status ON app_trips (status);
CREATE INDEX idx_app_trips_exit_time ON app_trips (exit_time DESC);

-- Dashboard-configurable RTSP cameras (vision worker + MJPEG read from here)
CREATE TABLE app_cameras (
    id               VARCHAR(64) PRIMARY KEY,
    name             VARCHAR(255) NOT NULL,
    gate_type        VARCHAR(16)  NOT NULL CHECK (gate_type IN ('entry', 'exit')),
    stream_url       TEXT         NOT NULL,
    status           VARCHAR(20)  NOT NULL DEFAULT 'offline'
        CHECK (status IN ('online', 'offline', 'error')),
    is_active        BOOLEAN      NOT NULL DEFAULT TRUE,
    last_heartbeat   TIMESTAMPTZ,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_app_cameras_active ON app_cameras (is_active);
CREATE INDEX idx_app_cameras_gate ON app_cameras (gate_type);

CREATE TRIGGER trigger_app_cameras_updated_at
    BEFORE UPDATE ON app_cameras
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
