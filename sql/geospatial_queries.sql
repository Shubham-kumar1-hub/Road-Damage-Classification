-- 1. Severe damage within 5 km of a zone center.
SELECT
    id,
    damage_type,
    severity,
    confidence,
    latitude,
    longitude,
    ST_Distance(
        location,
        ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)::geography
    ) / 1000.0 AS distance_km
FROM damage_reports
WHERE
    severity = 'high'
    AND ST_DWithin(
        location,
        ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326)::geography,
        5000
    )
ORDER BY distance_km ASC, confidence DESC;

-- 2. Priority repair queue.
SELECT
    id,
    damage_type,
    severity,
    confidence,
    latitude,
    longitude,
    created_at,
    CASE
        WHEN severity = 'high' THEN 3
        WHEN severity = 'medium' THEN 2
        WHEN severity = 'low' THEN 1
        ELSE 0
    END AS priority_score
FROM damage_reports
WHERE severity IN ('high', 'medium')
ORDER BY priority_score DESC, confidence DESC, created_at ASC;

-- 3. Damage count by approximate city grid.
SELECT
    ROUND(latitude::numeric, 2) AS lat_grid,
    ROUND(longitude::numeric, 2) AS lon_grid,
    damage_type,
    severity,
    COUNT(*) AS report_count
FROM damage_reports
GROUP BY lat_grid, lon_grid, damage_type, severity
ORDER BY report_count DESC;

-- 4. Recent pothole reports.
SELECT *
FROM damage_reports
WHERE
    damage_type = 'pothole'
    AND created_at >= NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;

