-- The declared column drop rules from DECISIONS.md D2 and D3, applied in SQL.
--
-- Thresholds are substituted from yield_screening/config.py, which is the executable
-- form of DECISIONS.md, so this file has no hard coded numbers to drift from it.
--
-- Rule order matters and is preserved: all missing, then constant, then near constant,
-- then the hard missing ceiling, then duplicates, then the dense/sparse band split.
-- The first rule that fires is the reason recorded, exactly as in the Python
-- implementation, so the two produce the same reason for the same sensor and not just
-- the same keep/drop verdict.

WITH stats AS (
    {sensor_stats}
),
ranked_duplicates AS (
    SELECT
        sensor_id,
        fingerprint,
        -- Keep the first sensor in column order within each identical group, matching
        -- the Python rule. Sensor names are zero padded, so lexical order is column
        -- order.
        ROW_NUMBER() OVER (PARTITION BY fingerprint ORDER BY sensor_id) AS rank_in_group
    FROM stats
),
decided AS (
    SELECT
        s.sensor_id,
        s.n_records,
        s.n_observed,
        s.n_missing,
        s.missing_rate,
        s.coverage,
        s.n_distinct_values,
        s.modal_share,
        s.fingerprint,
        d.rank_in_group,
        CASE
            WHEN s.n_observed = 0                            THEN 'all missing on train'
            WHEN s.n_distinct_values <= 1                    THEN 'constant on train'
            WHEN s.modal_share >= {modal_share_ceiling}      THEN 'near constant on train'
            WHEN s.missing_rate > {missing_hard_ceiling}
                THEN 'train missing rate above hard ceiling'
            WHEN d.rank_in_group > 1                         THEN 'duplicate of an earlier sensor'
            WHEN s.missing_rate > {dense_band_ceiling}
                THEN 'sparse band, values dropped, indicator retained'
            ELSE NULL
        END AS drop_reason
    FROM stats s
    JOIN ranked_duplicates d USING (sensor_id)
)
SELECT
    sensor_id,
    n_records,
    n_observed,
    n_missing,
    missing_rate,
    coverage,
    n_distinct_values,
    modal_share,
    fingerprint,
    drop_reason,
    drop_reason IS NULL AS is_dense_band,
    drop_reason = 'sparse band, values dropped, indicator retained' AS is_sparse_band
FROM decided
ORDER BY sensor_id;
