CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS damage_reports (
    id BIGSERIAL PRIMARY KEY,
    damage_type VARCHAR(80) NOT NULL,
    severity VARCHAR(30) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    image_path TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_damage_reports_location
    ON damage_reports
    USING GIST (location);

CREATE INDEX IF NOT EXISTS idx_damage_reports_severity
    ON damage_reports (severity);

CREATE INDEX IF NOT EXISTS idx_damage_reports_damage_type
    ON damage_reports (damage_type);

CREATE INDEX IF NOT EXISTS idx_damage_reports_created_at
    ON damage_reports (created_at DESC);

