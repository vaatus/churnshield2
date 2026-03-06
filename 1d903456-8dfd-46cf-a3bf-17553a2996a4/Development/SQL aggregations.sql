-- =====================================================
-- ZERVE SQL BLOCK: User Event Aggregations
-- Part of ChurnShield — Zerve Hackathon 2026
-- =====================================================
-- This SQL block runs on the events_clean DataFrame
-- passed from Block 1 (Load & Clean Data).
-- It demonstrates multi-language capability on Zerve.
-- =====================================================

SELECT
    user_id,

    -- Volume metrics
    COUNT(*) AS total_events,
    COUNT(DISTINCT event) AS unique_event_types,
    COUNT(DISTINCT date) AS active_days,
    COUNT(DISTINCT session_id_derived) AS total_sessions,

    -- Agent usage
    SUM(CASE WHEN event_category = 'agent_use' THEN 1 ELSE 0 END) AS agent_events,
    MAX(CASE WHEN event_category = 'agent_use' THEN 1 ELSE 0 END) AS used_agent,

    -- Code execution
    SUM(CASE WHEN event = 'run_block' THEN 1 ELSE 0 END) AS code_runs,
    MAX(CASE WHEN event = 'run_block' THEN 1 ELSE 0 END) AS ran_code,

    -- Creation events
    SUM(CASE WHEN event = 'canvas_create' THEN 1 ELSE 0 END) AS canvases_created,
    SUM(CASE WHEN event = 'block_create' THEN 1 ELSE 0 END) AS blocks_created,
    SUM(CASE WHEN event = 'edge_create' THEN 1 ELSE 0 END) AS edges_created,
    MAX(CASE WHEN event = 'edge_create' THEN 1 ELSE 0 END) AS has_edges,

    -- File operations
    SUM(CASE WHEN event = 'files_upload' THEN 1 ELSE 0 END) AS files_uploaded,

    -- Temporal bounds
    MIN(timestamp) AS first_event,
    MAX(timestamp) AS last_event,

    -- Tenure in hours
    ROUND(
        (JULIANDAY(MAX(timestamp)) - JULIANDAY(MIN(timestamp))) * 24.0,
        1
    ) AS tenure_hours,

    -- Session depth: avg events per session
    ROUND(
        CAST(COUNT(*) AS FLOAT) / NULLIF(COUNT(DISTINCT session_id_derived), 0),
        1
    ) AS events_per_session,

    -- Category diversity: number of distinct event categories used
    COUNT(DISTINCT event_category) AS category_diversity

FROM events_clean
WHERE user_id IS NOT NULL
GROUP BY user_id
HAVING COUNT(*) >= 3
ORDER BY total_events DESC
